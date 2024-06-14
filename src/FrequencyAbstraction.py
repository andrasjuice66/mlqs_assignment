import numpy as np
import pandas as pd

class FourierTransformation:
    
    def __init__(self):
        self.temp_list = []
        self.freqs = None

    def find_fft_transformation(self, data):
        transformation = np.fft.rfft(data, len(data))
        real_ampl = transformation.real

        if len(real_ampl) != len(self.freqs):
            real_ampl = real_ampl[:len(self.freqs)]  # Adjust to match lengths

        max_freq = self.freqs[np.argmax(real_ampl)]
        freq_weigthed = float(np.sum(self.freqs * real_ampl)) / np.sum(real_ampl)
        PSD = np.divide(np.square(real_ampl), float(len(real_ampl)))
        PSD_pdf = np.divide(PSD, np.sum(PSD))

        if np.count_nonzero(PSD_pdf) == PSD_pdf.size:
            pse = -np.sum(np.log(PSD_pdf) * PSD_pdf)
        else:
            pse = 0

        real_ampl = np.insert(real_ampl, 0, max_freq)
        real_ampl = np.insert(real_ampl, 0, freq_weigthed)
        row = np.insert(real_ampl, 0, pse)

        self.temp_list.append(row)

        return 0

    def abstract_frequency(self, data_table, columns, window_size, sampling_rate):
        self.freqs = (sampling_rate * np.fft.rfftfreq(int(window_size))).round(3)

        for col in columns:
            collist = []
            collist.append(col + '_max_freq')
            collist.append(col + '_freq_weighted')
            collist.append(col + '_pse')
            
            collist = collist + [col + '_freq_' +
                    str(freq) + '_Hz_ws_' + str(window_size) for freq in self.freqs]
           
            data_table[col].rolling(window_size + 1).apply(self.find_fft_transformation)

            frequencies = np.pad(np.array(self.temp_list), ((window_size, 0), (0, 0)),
                        'constant', constant_values=np.nan)

            data_table[collist] = pd.DataFrame(frequencies, index=data_table.index)

            del self.temp_list[:]
        
        return data_table
