import numpy as np
import librosa
from scipy.fft import dct
from scipy.signal import find_peaks, spectrogram


class SpeechFeatureExtractor:
    def __init__(self, audio_file):
        # 读取音频文件
        self.y, self.fs = librosa.load(audio_file, sr=None)
        self.features = self.extract_features()

    #添加随机白噪声，模拟现实环境
    def add_noise(self, feature_vector, noise_factor):
        """
        向特征向量添加噪声。

        :param feature_vector: numpy 数组，表示特征向量
        :param noise_factor: float，表示添加噪声的强度
        :return: 带噪声的特征向量
        """
        noise = np.random.randn(len(feature_vector))
        augmented_data = feature_vector + noise_factor * noise
        return augmented_data

    def enframe(self, y, win, hop):
        """
                将信号分帧，并应用窗函数。
                :param y: 音频信号
                :param win: 窗函数
                :param hop: 每帧的间隔
                :return: 分帧后的信号
                """
        num_frames = 1 + int((len(y) - len(win)) / hop)
        frames = np.zeros((num_frames, len(win)))

        for i in range(num_frames):
            frames[i, :] = y[i * hop: i * hop + len(win)] * win

        return frames

    #计算MEL滤波器组
    def melbankm(self, p, n, fs, fl=0, fh=0.5, w='M'):
        """
        计算Mel滤波器组。
        :param p: 过滤器组的数量
        :param n: FFT长度
        :param fs: 采样率
        :param fl: 最低频率
        :param fh: 最高频率
        :param w: 窗函数类型（'M'、'N' 或 'R'）
        :return: 滤波器组矩阵、滤波器组下标的起始值、滤波器组下标的结束值
        """
        fmelmax = 1127 * np.log(1 + fh * fs / 700)  # Convert Hz to Mel
        fmelmin = 1127 * np.log(1 + fl * fs / 700)  # Convert Hz to Mel

        # Generate p linearly spaced points between fmelmin and fmelmax
        melpoints = np.linspace(fmelmin, fmelmax, p + 2)

        # Convert Mel to Hz
        f = 700 * (np.exp(melpoints / 1127) - 1)

        # Round f to the nearest FFT bin
        fbin = np.floor((n + 1) * f / fs).astype(int)

        # Create the filter bank matrix
        melbank = np.zeros((p, n))

        for k in range(1, p + 1):
            for i in range(fbin[k - 1], fbin[k]):
                melbank[k - 1, i] = (i - fbin[k - 1]) / (fbin[k] - fbin[k - 1])
            for i in range(fbin[k], fbin[k + 1]):
                melbank[k - 1, i] = (fbin[k + 1] - i) / (fbin[k + 1] - fbin[k])

        return melbank, fbin[0], fbin[p + 1]

    ###########快速傅立叶变换
    def custom_rfft(self, signal, n=None):
        """
        计算实数输入的快速傅立叶变化
        ：signal：输入矩阵
        ：return：输出矩阵
        """
        return np.fft.rfft(signal, n)
    ###########
    #实值离散余弦变换（Real Discrete Cosine Transform）的计算
    def rdct(self, x, norm=None):
        return dct(x, type=2, norm=norm)

    #计算MEL倒谱系数
    def melcepst(self, s, fs, w='M', nc=12, p=None, n=None, inc=None, fl=0, fh=0.5):
        """
        计算Mel倒谱系数。
        :param s: 输入信号
        :param fs: 采样率
        :param w: 窗函数类型（'M'、'N' 或 'R'）
        :param nc: 倒谱系数的数量
        :param p: 过滤器组的数量
        :param n: 帧长度
        :param inc: 帧移
        :param fl: 最低频率
        :param fh: 最高频率
        :return: Mel倒谱系数
        """
        if p is None:
            p = int(3 * np.log(fs))
        if n is None:
            n = int(2 ** (np.floor(np.log2(0.03 * fs))))
        if inc is None:
            inc = int(n / 2)

        if w == 'R':
            z = self.enframe(s, n, inc)
        elif w == 'N':
            z = self.enframe(s, np.hanning(n), inc)
        else:
            z = self.enframe(s, np.hamming(n), inc)

        f = self.custom_rfft(z.T, n=n)
        m, a, b = self.melbankm(p, n, fs)
        pw = f[a:b, :] * np.conj(f[a:b, :])
        pth = np.max(pw) * 1e-6

        if w == 'p':
            y = np.log(np.maximum(m @ pw, pth))
        else:
            ath = np.sqrt(pth)

            y = np.log(np.maximum(m @ np.abs(f[:, a:b]), ath))

        y = y.real
        c = self.rdct(y).T
        nf, nc = c.shape

        if p > nc:
            c = np.hstack((c, np.zeros((nf, nc - p))))
        elif p < nc:
            c = c[:, :nc]

        if not w == '0':
            c = c[:, 1:]

        if w == 'e':
            c = np.hstack((np.log(np.sum(pw)).reshape(-1, 1), c))

        # 计算导数
        if w == 'D':
            vf = np.array([4, 3, 2, 1, 0, -1, -2, -3, -4]) / 60
            af = np.array([1, -1]) / 2
            ww = np.ones(5)
            cx = np.row_stack((c[np.newaxis, ww.astype(int), :], c, c[np.newaxis, -ww.astype(int), :]))
            vx = np.reshape(np.convolve(cx.ravel(), vf, mode='same'), (nf + 10, nc))
            vx = vx[8:, :]
            ax = np.reshape(np.convolve(vx.ravel(), af, mode='same'), (nf + 2, nc))
            ax = ax[2:, :]
            vx = vx[:-2, :]

            if w == 'd':
                c = np.column_stack((c, vx, ax))
            else:
                c = np.column_stack((c, ax))
        elif w == 'd':
            vf = np.array([4, 3, 2, 1, 0, -1, -2, -3, -4]) / 60
            ww = np.ones(4)
            cx = np.row_stack((c[np.newaxis, ww.astype(int), :], c, c[np.newaxis, -ww.astype(int), :]))
            vx = np.reshape(np.convolve(cx.ravel(), vf, mode='same'), (nf + 8, nc))
            vx = vx[8:, :]
            c = np.column_stack((c, vx))

        return c

    #信号在0-250 Hz频率范围内的能量比例
    def energy_ratio(self, freq_range=(0, 250)):
        f, t, Sxx = spectrogram(self.y, self.fs)
        sn = 20 * np.log10(np.abs(Sxx) + np.finfo(float).eps)
        sn1 = sn + np.min(sn)

        max_freq = np.max(f)
        n = int(np.round(len(f) * freq_range[1] / max_freq))

        Eratio = np.sum(sn1[:n, :]) / np.sum(sn1)

        return Eratio

    def extract_fundamental_frequency(self, samplerate, frame_length=1024, hop_length=512):
        '''
        提取信号帧的基频。
        speech : array_like
            输入信号。
        samplerate : int
            采样率。
        frame_length : int, 可选
            帧长度。默认为1024。
        hop_length : int, 可选
            帧间距。默认为512。
        '''
        # 计算声谱图
        speech = self.y
        spectrogram = np.abs(librosa.stft(speech, n_fft=frame_length, hop_length=hop_length))

        # 使用piptrack方法提取基频
        pitches, magnitudes = librosa.piptrack(S=spectrogram, sr=samplerate)

        # 提取最大幅度的频率作为基频
        f0 = []
        for t in range(pitches.shape[1]):
            index = np.argmax(magnitudes[:, t])
            f0.append(pitches[index, t])

        return np.array(f0)

    def MFCC(self):
        # Extract MFCC features
        n_mfcc = 13  # Number of MFCCs to extract
        mfccs = librosa.feature.mfcc(y=self.y, sr=self.fs, n_mfcc=n_mfcc)

        #normalize the MFCCs
        #mfccs = librosa.util.normalize(mfccs, axis=1)
        return mfccs

    #计算MFCC一阶差分
    def MFCC_diff(self, mfcc, n_mfcc=13):
        mfcc_delta = librosa.feature.delta(mfcc)
        return  mfcc_delta

    def formant_jitter(self, Fm):
        x1 = 0
        for i in range(len(Fm) - 1):
            t1 = abs(Fm[i] - Fm[i + 1])
            x1 = x1 + t1
        Fm_Jitter1 = 100 * x1 / (np.mean(Fm) * (len(Fm) - 1))
        return Fm_Jitter1

    #二阶基音频率抖动
    def frequency_jitter2(self,Fm):
        nframe = len(Fm)
        x = 0

        for i in range(1, nframe - 1):
            t = abs(2 * Fm[i] - Fm[i + 1] - Fm[i - 1])
            x += t

        F_Jitter2 = 100 * x / (np.mean(Fm) * (nframe - 2))

        return F_Jitter2

    #浊音帧基频的差分
    def voiced_frame_diff(self, F):
        nframe = len(F)
        dF = []
        for i in range(nframe - 1):
            if F[i] * F[i + 1] != 0:
                dF.append(F[i] - F[i + 1])

        return np.array(dF)

    def mapzo(self, x):
        #输入向量x的元素映射到0和1之间
        xmin = np.min(x)
        xmax = np.max(x)
        m = (x - xmin) / (xmax - xmin)
        return m

    def voicemark(self, speech):
        # 在此处实现voicemark函数的代码...
        # 假设已经计算出了begin和last
        begin = 0
        last = len(speech) - 1

        return begin, last

    def pick_peak(self, speech, samplerate):
        # 在此处实现pick_peak函数的代码...

        # 预处理：计算语音信号的功率谱
        spectrum = np.abs(np.fft.fft(speech))
        frequencies = np.fft.fftfreq(len(speech), 1 / samplerate)

        # 使用find_peaks函数找到频率谱中的峰值
        peaks, _ = find_peaks(spectrum)
        peak_frequencies = frequencies[peaks]
        peak_spectrum = spectrum[peaks]

        # 计算共振峰频率矩阵和带宽矩阵
        s = np.array(peak_frequencies)  # 示例：共振峰频率矩阵
        s_bw = np.array(peak_spectrum)  # 示例：共振峰带宽矩阵

        return s, s_bw, None

    def smoothing(self, s, s_bw, window_size=3):
        # 在此处实现smoothing函数的代码...
        def moving_average(data, window_size):
            return np.convolve(data, np.ones(window_size) / window_size, mode='same')

        formant_frame = np.apply_along_axis(moving_average, axis=0, arr=s, window_size=window_size)
        bw_frame = np.apply_along_axis(moving_average, axis=0, arr=s_bw, window_size=window_size)

        return formant_frame, bw_frame

    def formant_get(self, speech, samplerate):
        M = 160  # 帧移
        # [speech, samplerate] = audioread('e.wav')
        [begin, last] = self.voicemark(speech)
        # 提取共振峰频率和带宽
        [s, s_bw, _] = self.pick_peak(speech, samplerate)
        # 对频率和带宽进行平滑
        [formant_frame, bw_frame] = self.smoothing(s, s_bw)

        # 开始对共振峰进行处理，静音段置0
        frame_num = formant_frame.shape[0]
        frame_begin = 0  # 起始帧
        frame_last = 0  # 结束帧

        # 确定起始帧
        for i in range(frame_num):
            if begin <= (i - 1) * M + 1:
                break
            frame_begin += 1

        # 确定结束帧
        for i in range(frame_num):
            if last <= (i - 1) * M + 1:
                break
            frame_last += 1

        # 根据起始帧和结束帧对共振峰数据进行处理

        for i in range(frame_begin - 1):
            formant_frame[i] = 0
        for i in range(frame_last, frame_num):
            formant_frame[i] = 0

        #共振峰频率矩阵formant_frame和带宽矩阵bw_frame
        return formant_frame, bw_frame

    def calculate_energy(self,y,win,hop):
        frames = self.enframe(y, win, hop)
        energy = np.sum(frames ** 2, axis=1)
        return energy

    def calculate_shimmer(self, energy):
        shimmer = 0
        nframe = len(energy)
        for i in range(1, nframe):
            shimmer += abs(energy[i] - energy[i - 1]) / nframe
        return 100 * shimmer / np.mean(energy)

    def calculate_reg_coff(self, energy):
        nframe = len(energy)
        x = np.arange(nframe)
        y = energy - np.mean(energy)
        s1 = np.sum(x * y)
        s2 = np.sum(x ** 2)
        x4 = np.sum(x ** 4)
        reg_coff = (s1 * s2 * nframe) / (s2 ** 2 - x4 * nframe)
        return reg_coff

    def calculate_sqr_err(self, energy, reg_coff):
        nframe = len(energy)
        x = 0
        for i in range(nframe):
            t = energy[i] - (np.mean(energy) - reg_coff * (i + 1) / nframe)
            x += t ** 2 / nframe
        sqr_err = x
        return sqr_err

    def extract_features(self):
        y = self.y
        fs = self.fs

        y = self.add_noise(y, noise_factor=0.01)
        # Parameters
        win = np.hamming(400)  # Window of 400 samples
        hop = 200  # Hop size of 200 samples

        # Calculate short-time energy
        energy = self.calculate_energy(y, win, hop)

        # 计算共振峰频率和带宽
        formant_frame, bw_frame = self.formant_get(y, samplerate=8000)

        # 从共振峰矩阵中提取Fm1
        Fm1 = formant_frame

        # 计算共振峰抖动
        Fm_Jitter1 = self.formant_jitter(Fm1)

        #计算基频
        Fund_freq = self.extract_fundamental_frequency(samplerate = 8000)
        F_max = np.max(Fund_freq)
        F_min = np.min(Fund_freq)
        F_mean = np.mean(Fund_freq)
        F_var = np.var(Fund_freq)

        #二阶音频抖动
        Fm_Jitter2 = self.frequency_jitter2(Fund_freq)

        #浊音帧基频的差分
        Voice_diff = self.voiced_frame_diff(Fund_freq)
        V_max = np.max(Voice_diff)
        V_min = np.min(Voice_diff)
        V_mean = np.mean(Voice_diff)
        V_var = np.var(Voice_diff)

        #计算能量
        E_max = np.max(energy)
        E_min = np.min(energy)
        E_mean = np.mean(energy)
        E_var = np.var(energy)
        E_ratio = self.energy_ratio()
        E_shimmer = self.calculate_shimmer(energy)
        E_reg_coff = self.calculate_reg_coff(energy)
        E_sqr_err = self.calculate_sqr_err(energy, E_reg_coff)

        # 计算MFCC
        E_MFCC = self.MFCC()
        max_mfccs = np.max(E_MFCC, axis=1)
        min_mfccs = np.min(E_MFCC, axis=1)
        mean_mfccs = np.mean(E_MFCC, axis=1)
        var_mfccs = np.var(E_MFCC, axis=1)

        #计算MFCC一阶差分
        MFCC_diff= self.MFCC_diff(mfcc=E_MFCC)
        Md_max = np.max(MFCC_diff, axis=1)
        Md_min = np.min(MFCC_diff, axis=1)
        Md_mean = np.mean(MFCC_diff, axis=1)
        Md_var = np.var(MFCC_diff, axis=1)
        ##########
        #np.set_printoptions(threshold=np.inf)

        # Combine features into a single array or list
        features = np.array([Fm_Jitter1, Fm_Jitter2, F_max, F_min, F_mean, F_var, V_max, V_min, V_mean, V_var, E_max, E_min, E_mean, E_var, E_ratio, E_shimmer, E_reg_coff, E_sqr_err])
        features = np.concatenate((features, max_mfccs, min_mfccs, mean_mfccs, var_mfccs, Md_max, Md_min, Md_mean, Md_var), axis=0)
        features = self.mapzo(features)
        return features






