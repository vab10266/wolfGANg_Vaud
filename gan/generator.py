import torch
from torch import nn
from .temp_network import TemporalNetwork
from .bar_generator import BarGenerator


class MuseGenerator(nn.Module):
    
    n_tracks = 4
    n_bars = 2
    n_steps_per_bar = 16
    n_pitches = 84
    
    def __init__(self,
                 c_dimension=10,
                 z_dimension=32,
                 hid_channels=1024,
                 hid_features=1024,
                 out_channels=1):
        super().__init__()
        # chords generator
        self.c_chords_network = TemporalNetwork(dimension=1,
                                              hid_channels=hid_channels)
        self.z_chords_network = TemporalNetwork(dimension=z_dimension,
                                              hid_channels=hid_channels)
        # melody generators
        self.c_melody_networks = nn.ModuleDict({})
        for n in range(self.n_tracks):
            self.c_melody_networks.add_module('c_melodygen_'+str(n),
                                            TemporalNetwork(dimension=1,
                                                            hid_channels=hid_channels))
        self.z_melody_networks = nn.ModuleDict({})
        for n in range(self.n_tracks):
            self.z_melody_networks.add_module('z_melodygen_'+str(n),
                                            TemporalNetwork(dimension=z_dimension,
                                                            hid_channels=hid_channels))
        # bar generators
        self.bar_generators = nn.ModuleDict({})   
        for n in range(self.n_tracks):
            self.bar_generators.add_module('bargen_'+str(n),
                                           BarGenerator(c_dimension=4,
                                                        z_dimension=4*z_dimension,
                                                        hid_features=hid_features,
                                                        hid_channels=hid_channels//2,
                                                        out_channels=out_channels))
        # musegan generator compiled
        
    def forward(self, c, z):
        # print("Generator:")
        # print("c: ", *c.shape, c.dtype)
        # c shape: (batch_size, c_dimension)
        c_chords = c[:, 0].reshape((-1, 1, 1))
        c_style = c[:, 1].reshape((-1, 1))
        c_melody = c[:, 2:2 + self.n_tracks].reshape((-1, self.n_tracks, 1))
        c_groove = c[:, 2 + self.n_tracks:].reshape((-1, self.n_tracks, 1))
        
        
        # z shape: (batch_size, 2*n_tracks + 2, z_dimension)
        z_chords = z[:, 0, :]
        z_style = z[:, 1, :]
        z_melody = z[:, 2:2 + self.n_tracks, :]
        z_groove = z[:, 2 + self.n_tracks:, :]

        # chords shape: (batch_size, _dimension)
        # style shape: (batch_size, _dimension)
        # melody shape: (batch_size, n_tracks, _dimension)
        # groove shape: (batch_size, n_tracks, _dimension)
        c_chord_outs = self.c_chords_network(c_chords)
        z_chord_outs = self.z_chords_network(z_chords)
        bar_outs = []
        for bar in range(self.n_bars):
            track_outs = []
            c_chord_out = c_chord_outs[:, :, bar]
            z_chord_out = z_chord_outs[:, :, bar]
            c_style_out = c_style
            z_style_out = z_style
            for track in range(self.n_tracks):
                c_melody_in = c_melody[:, track]
                z_melody_in = z_melody[:, track]
                c_melody_out = self.c_melody_networks['c_melodygen_'+str(track)](c_melody_in)[:, :, bar]
                z_melody_out = self.z_melody_networks['z_melodygen_'+str(track)](z_melody_in)[:, :, bar]
                c_groove_out = c_groove[:, track, :]
                z_groove_out = z_groove[:, track, :]
                # print(c_chord_out.shape, c_style_out.shape, c_melody_out.shape, c_groove_out.shape)
                c = torch.cat([c_chord_out, c_style_out, c_melody_out, c_groove_out], dim=1)
                z = torch.cat([z_chord_out, z_style_out, z_melody_out, z_groove_out], dim=1)
                track_outs.append(self.bar_generators['bargen_'+str(track)](c, z))
                # output shape: (batch_size, out_channels, 1, n_steps_per_bar, n_pitches)
            track_out = torch.cat(track_outs, dim=1)
            bar_outs.append(track_out)
        out = torch.cat(bar_outs, dim=2)
        # out shape: (batch_size, n_tracks, n_bars, n_steps_per_bar, n_pitches)
        return out
