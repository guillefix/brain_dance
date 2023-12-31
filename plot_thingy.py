# -*- coding: utf-8 -*-
# vispy: gallery 2
# Copyright (c) 2015, Vispy Development Team.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

## TAKEN FROM https://github.com/alexandrebarachant/muse-lsl basically

"""
Multiple real-time digital signals with GLSL-based clipping.
"""

from vispy import gloo, app, visuals

import numpy as np
import math
from seaborn import color_palette
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import lfilter, lfilter_zi
from mne.filter import create_filter
from constants import LSL_SCAN_TIMEOUT, LSL_EEG_CHUNK
import mne_connectivity

import utils

BUFFER_LENGTH = 5

# Length of the epochs used to compute the FFT (in seconds)
EPOCH_LENGTH = 1

# Amount of overlap between two consecutive epochs (in seconds)
OVERLAP_LENGTH = 0.8

# Amount to 'shift' the start of each next consecutive epoch
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

# Index of the channel(s) (electrodes) to be used
# 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
# INDEX_CHANNEL = [0]
INDEX_CHANNEL = [3]

class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3


VERT_SHADER = """
#version 120
// y coordinate of the position.
attribute float a_position;
// row, col, and time index.
attribute vec3 a_index;
varying vec3 v_index;
// 2D scaling factor (zooming).
uniform vec2 u_scale;
// Size of the table.
uniform vec2 u_size;
// Number of samples per signal.
uniform float u_n;
// Color.
attribute vec3 a_color;
varying vec4 v_color;
// Varying variables used for clipping in the fragment shader.
varying vec2 v_position;
varying vec4 v_ab;
void main() {
    float n_rows = u_size.x;
    float n_cols = u_size.y;
    // Compute the x coordinate from the time index.
    float x = -1 + 2*a_index.z / (u_n-1);
    vec2 position = vec2(x - (1 - 1 / u_scale.x), a_position);
    // Find the affine transformation for the subplots.
    vec2 a = vec2(1./n_cols, 1./n_rows)*.9;
    vec2 b = vec2(-1 + 2*(a_index.x+.5) / n_cols,
                    -1 + 2*(a_index.y+.5) / n_rows);
    // Apply the static subplot transformation + scaling.
    gl_Position = vec4(a*u_scale*position+b, 0.0, 1.0);
    v_color = vec4(a_color, 1.);
    v_index = a_index;
    // For clipping test in the fragment shader.
    v_position = gl_Position.xy;
    v_ab = vec4(a, b);
}
"""

FRAG_SHADER = """
#version 120
varying vec4 v_color;
varying vec3 v_index;
varying vec2 v_position;
varying vec4 v_ab;
void main() {
    gl_FragColor = v_color;
    // Discard the fragments between the signals (emulate glMultiDrawArrays).
    if ((fract(v_index.x) > 0.) || (fract(v_index.y) > 0.))
        discard;
    // Clipping test.
    vec2 test = abs((v_position.xy-v_ab.zw)/v_ab.xy);
    if ((test.x > 1))
        discard;
}
"""


def view():
    print("Looking for an EEG stream...")
    streams = resolve_byprop('type', 'EEG', timeout=LSL_SCAN_TIMEOUT)

    if len(streams) == 0:
        raise(RuntimeError("Can't find EEG stream."))
    print("Start acquiring data.")

    inlet = StreamInlet(streams[0], max_chunklen=LSL_EEG_CHUNK)
    # inlet2 = StreamInlet(streams[1], max_chunklen=LSL_EEG_CHUNK)
    Canvas([inlet])
    # Canvas([inlet, inlet])
    app.run()


class Canvas(app.Canvas):
    def __init__(self, lsl_inlets, scale=500, filt=True):
        app.Canvas.__init__(self, title='EEG - Use your wheel to zoom!',
                            keys='interactive')

        self.inlets = lsl_inlets
        info = self.inlets[0].info()
        description = info.desc()

        window = 10
        self.sfreq = info.nominal_srate()
        n_samples = int(self.sfreq * window)
        self.n_chans = info.channel_count()
        # self.n_chans = info.channel_count() + 1

        self.n_metrics = 3
        self.n_feats = self.n_chans + self.n_metrics

        ch = description.child('channels').first_child()
        ch_names = [ch.child_value('label')]

        # print(self.n_chans)
        for i in range(self.n_chans - 1):
            ch = ch.next_sibling()
            ch_names.append(ch.child_value('label'))

        ch_names = ch_names[::-1]

        # for i in range(self.n_metrics):
        #     ch_names.append("Feat")
        ch_names += ["Alpha", "Beta", "Theta"]

        print(ch_names)

        # Number of cols and rows in the table.
        n_rows = self.n_feats
        n_cols = 1

        # Number of signals.
        m = n_rows * n_cols

        # Number of samples per signal.
        n = n_samples

        # Various signal amplitudes.
        amplitudes = np.zeros((m, n)).astype(np.float32)
        # gamma = np.ones((m, n)).astype(np.float32)
        # Generate the signals as a (m, n) array.
        y = amplitudes

        color = color_palette("RdBu_r", n_rows)

        color = np.repeat(color, n, axis=0).astype(np.float32)
        # Signal 2D index of each vertex (row and col) and x-index (sample index
        # within each signal).
        index = np.c_[np.repeat(np.repeat(np.arange(n_cols), n_rows), n),
                      np.repeat(np.tile(np.arange(n_rows), n_cols), n),
                      np.tile(np.arange(n), m)].astype(np.float32)

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program['a_position'] = y.reshape(-1, 1)
        self.program['a_color'] = color
        self.program['a_index'] = index
        self.program['u_scale'] = (1., 1.)
        self.program['u_size'] = (n_rows, n_cols)
        self.program['u_n'] = n


        info = self.inlets[0].info()
        self.fs = int(info.nominal_srate())
        n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                                  SHIFT_LENGTH + 1))

        # Initialize the band power buffer (for plotting)
        # bands will be ordered: [delta, theta, alpha, beta]
        self.eeg_buffers = []
        self.band_buffers = []
        self.filter_states = []
        for inlet in self.inlets:
            eeg_buffer = np.zeros((int(self.fs * BUFFER_LENGTH), 1))
            filter_state = None  # for use with the notch filter
            band_buffer = np.zeros((n_win_test, 4))
            self.eeg_buffers.append(eeg_buffer)
            self.filter_states.append(filter_state)
            self.band_buffers.append(band_buffer)

        # text
        self.font_size = 48.
        self.names = []
        self.quality = []
        for ii in range(len(ch_names)):
            text = visuals.TextVisual(ch_names[ii], bold=True, color='white')
            # print(text.text)
            self.names.append(text)
            # self.names[ii].pos *= np.array([0,0.5,0])
            # self.names[ii].pos += np.array([0,-100000,0])
            text = visuals.TextVisual('', bold=True, color='white')
            self.quality.append(text)

        self.quality_colors = color_palette("RdYlGn", 11)[::-1]

        self.scale = np.array([1 for i in range(self.n_feats)])
        # self.scale = scale
        # self.scale = np.expand_dims(self.scale, 0)
        self.scale[-1] = 1
        self.scale[-2] = 1000
        self.scale[-3] = 1000
        self.scale[-4] = 1000
        self.scale[-5] = 1000

        self.scale[1] = 2
        self.scale[0] = 2
        self.n_samples = n_samples
        self.filt = filt
        self.af = [1.0]

        # self.data_f = np.zeros((n_samples, self.n_chans))

        self.data_fs = []
        self.bfs = []
        self.filt_states = []
        for i in range(len(self.inlets)):
            self.data_f = np.zeros((n_samples, self.n_feats))
            self.data_fs.append(self.data_f)
            # self.data = np.zeros((n_samples, self.n_chans)) # minus number of bands

            self.bf = create_filter(self.data_f.T, self.sfreq, 3, 40.,
                                    method='fir')
            self.bfs.append(self.bf)

            zi = lfilter_zi(self.bf, self.af)
            self.filt_state = np.tile(zi, (self.n_chans, 1)).transpose()
            self.filt_states.append(self.filt_state)

        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        gloo.set_viewport(0, 0, *self.physical_size)
        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

        self.show()

    def on_key_press(self, event):

        # toggle filtering
        if event.key.name == 'D':
            self.filt = not self.filt

        # increase time scale
        if event.key.name in ['+', '-']:
            if event.key.name == '+':
                dx = -0.05
            else:
                dx = 0.05
            scale_x, scale_y = self.program['u_scale']
            scale_x_new, scale_y_new = (scale_x * math.exp(1.0 * dx),
                                        scale_y * math.exp(0.0 * dx))
            self.program['u_scale'] = (
                max(1, scale_x_new), max(1, scale_y_new))
            self.update()

    def on_mouse_wheel(self, event):
        dx = np.sign(event.delta[1]) * .05
        scale_x, scale_y = self.program['u_scale']
        scale_x_new, scale_y_new = (scale_x * math.exp(0.0 * dx),
                                    scale_y * math.exp(2.0 * dx))
        self.program['u_scale'] = (max(1, scale_x_new), max(0.01, scale_y_new))
        self.update()

    def on_timer(self, event):
        """Add some data at the end of each signal (real-time signals)."""

        corr_metrics = []

        for inlet_index, inlet in enumerate(self.inlets):
            # if inlet_index == 0:
            samples, timestamps = inlet.pull_chunk(timeout=1,
                                                        max_samples=int(SHIFT_LENGTH * self.fs))
            if timestamps:
                # samples = np.array(samples)[:, ::-1]
                # print(samples.shape)

                # print(self.data.shape)
                # print(samples.shape)

                # Only keep the channel we're interested in
                ch_data = np.array(samples)[:, INDEX_CHANNEL]
                # print(ch_data.shape)

                # Update EEG buffer with the new data
                self.eeg_buffers[inlet_index], self.filter_states[inlet_index] = utils.update_buffer(
                    self.eeg_buffers[inlet_index], ch_data, notch=True,
                    filter_state=self.filter_states[inlet_index])

                """ 3.2 COMPUTE BAND POWERS """
                # Get newest samples from the buffer
                data_epoch = utils.get_last_data(self.eeg_buffers[inlet_index],
                                                 EPOCH_LENGTH * self.fs)

                band_powers = utils.compute_band_powers(data_epoch, self.fs)
                self.band_buffers[inlet_index], _ = utils.update_buffer(self.band_buffers[inlet_index],
                                                     np.asarray([band_powers]))
                smooth_band_powers = np.mean(self.band_buffers[inlet_index], axis=0)
                # alpha_metric = smooth_band_powers[Band.Alpha] / \
                #     smooth_band_powers[Band.Delta]
                alpha_metric = smooth_band_powers[Band.Alpha]
                    # smooth_band_powers[Band.Delta]
                # print(alpha_metric)
                # beta_metric = smooth_band_powers[Band.Beta] / \
                #     smooth_band_powers[Band.Theta]
                beta_metric = smooth_band_powers[Band.Beta]
                # print(beta_metric)

                # theta_metric = smooth_band_powers[Band.Theta] / \
                #     smooth_band_powers[Band.Alpha]
                theta_metric = smooth_band_powers[Band.Theta]
                print(theta_metric)


                filt_samples, self.filt_states[inlet_index] = lfilter(self.bfs[inlet_index], self.af, samples,
                                                        axis=0, zi=self.filt_states[inlet_index])

                filt_samples_ext = np.concatenate([np.tile(alpha_metric, (filt_samples.shape[0], 1)), filt_samples], axis=-1)
                filt_samples_ext = np.concatenate([np.tile(beta_metric, (filt_samples.shape[0], 1)), filt_samples_ext], axis=-1)
                filt_samples_ext = np.concatenate([np.tile(theta_metric, (filt_samples.shape[0], 1)), filt_samples_ext], axis=-1)

                self.data_fs[inlet_index] = np.vstack([self.data_fs[inlet_index], filt_samples_ext])
                self.data_fs[inlet_index] = self.data_fs[inlet_index][-self.n_samples:]
                corr_metrics.append(self.data_fs[inlet_index][:,3])



                if inlet_index != 0: continue
                #plot

                # plot_data = self.data_f / self.scale
                mean = np.mean(self.data_fs[0], axis=0, keepdims=True)
                std = np.std(self.data_fs[0], axis=0, keepdims=True)
                plot_data = (self.data_fs[0] - mean)/self.scale

                for ii in range(len(self.names)):
                    index = plot_data.shape[1]-1-ii
                    self.quality[ii].text = '%.2f' % (self.data_fs[0][-1,index]/self.scale[index])
                    self.quality[ii].pos = np.array([self.physical_size[1]-100,20+80*ii])
                    # self.quality[ii].color = self.quality_colors[co[ii]]
                    # self.quality[ii].font_size = 12 + co[ii]

                    # self.names[ii].font_size = 12 + co[ii]
                    self.names[ii].pos = np.array([100,20+80*ii])
                    # self.names[ii].color = self.quality_colors[co[ii]]

                self.program['a_position'].set_data(
                    plot_data.T.ravel().astype(np.float32))
                self.update()

        corr_metrics_arr = np.stack(corr_metrics)
        corrcoefs = np.corrcoef(corr_metrics_arr)
        # print(corr_metrics_arr.shape)
        phase_lock = mne_connectivity.spectral_connectivity_time(np.expand_dims(corr_metrics_arr,0), method='plv', sfreq=self.sfreq, fmin=8, fmax=30, freqs=30)
        # print(phase_lock.shape)

        # print(corr_metrics.shape)
        if corr_metrics_arr.shape[0] > 1:
            corrcoef = corrcoefs[0,1]
            print("corrcoef: "+str(corrcoef))
            # print("phase_lock: "+str(phase_lock.get_data()))
            print("phase_lock: "+str(phase_lock.get_data()[0,2,0]))

    def on_resize(self, event):
        # Set canvas viewport and reconfigure visual transforms to match.
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)

        for ii, t in enumerate(self.names):
            t.transforms.configure(canvas=self, viewport=vp)
            t.pos = (self.size[0] * 0.025,
                     ((ii + 0.5) / self.n_chans) * self.size[1])

        for ii, t in enumerate(self.quality):
            t.transforms.configure(canvas=self, viewport=vp)
            t.pos = (self.size[0] * 0.975,
                     ((ii + 0.5) / self.n_chans) * self.size[1])

    def on_draw(self, event):
        gloo.clear()
        gloo.set_viewport(0, 0, *self.physical_size)
        # gloo.set_viewport(-1, -1, self.physical_size[0]*0.9, self.physical_size[1]*0.9)
        self.program.draw('line_strip')
        [t.draw() for t in self.names + self.quality]


view()
