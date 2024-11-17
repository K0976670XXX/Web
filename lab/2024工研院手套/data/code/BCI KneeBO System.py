import numpy as np
from scipy import signal
from sklearn.metrics import accuracy_score
import mne
import joblib

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import socket
from pylsl import StreamInlet, resolve_stream
from pylsl import StreamInfo, StreamOutlet

import os
import subprocess
import time


def BCI_KneeBO(event_stream_name='BCI - LSL Marker', kneebo_server_ip='127.0.0.6', kneebo_server_port=56950):
    ################################################################################
    # Figure - Loading
    # Experiment Ready/Start/End Figures - Loading
    global click
    click = False
    ready_pic = plt.imread('exp_img/ready_pic.jpg')
    fixation_pic = plt.imread('exp_img/fixation_pic.png')
    start_pic = plt.imread('exp_img/start_pic.jpg')
    end_pic = plt.imread('exp_img/end_pic.jpg')

    # Experiment Countdown Figures - Loading
    filedir = 'exp_img'
    filesubstr = 'init_count'
    filelist = [filedir+'/'+_ for _ in os.listdir(filedir) if filesubstr in _]
    filelist.sort(reverse=True)
    init_count_pic_list = [plt.imread(_) for _ in filelist]

    # Trial Countdown Figures - Loading
    filesubstr = 'task_count'
    filelist = [filedir+'/'+_ for _ in os.listdir(filedir) if filesubstr in _]
    filelist.sort(reverse=True)
    task_count_pic_list = [plt.imread(_) for _ in filelist]

    # Correct/Incorrect Feedback Figures - Loading
    true_pic = plt.imread('exp_img/true_pic.jpg')
    false_pic = plt.imread('exp_img/false_pic.jpg')
    late_pic = plt.imread('exp_img/late_pic.jpg')
    noisy_pic = plt.imread('exp_img/noisy_pic.png')

    ################################################################################
    # Experiment Scene - Initialization
    # Training/Testing Phase Selection
    while True:
        phase_select = input('Please input the phase of system:\n1) Training Phase\n2) Testing Phase\n')
        try:
            if int(phase_select) == 1:
                phase_select = 'Train'

                # Action Cue Figures - Loading
                cue_pic_dict = dict()
                cue_pic_dict.update({51: plt.imread('exp_img/rest_cue_pic.jpg')})
                cue_pic_dict.update({41: plt.imread('exp_img/left_cue_pic.jpg')})
                cue_pic_dict.update({61: plt.imread('exp_img/right_cue_pic.jpg')})

                print('\n'+'#'*50)
                break
            elif int(phase_select) == 2:
                phase_select = 'Test'
                print('\n'+'#'*50)
                break
        except:
            pass
        print('Input error, please try again!!')
        print('\n'+'#'*50)

    if phase_select == 'Test':
        while True:
            # Left/Right Leg Selection
            while True:
                leg_select = input('Please select the side of leg:\n1) Left Leg\n2) Right Leg\n')
                try:
                    if int(leg_select) == 1:
                        leg_name = 'l'

                        # Action Cue Figures - Loading
                        cue_pic_dict = dict()
                        cue_pic_dict.update({51: plt.imread('exp_img/rest_cue_pic.jpg')})
                        cue_pic_dict.update({41: plt.imread('exp_img/left_cue_pic.jpg')})

                        leg_select = 42
                        y_label = [41, 51]
                        print('\n'+'#'*50)
                        break
                    elif int(leg_select) == 2:
                        leg_name = 'r'

                        # Action Cue Figures - Loading
                        cue_pic_dict = dict()
                        cue_pic_dict.update({51: plt.imread('exp_img/rest_cue_pic.jpg')})
                        cue_pic_dict.update({61: plt.imread('exp_img/right_cue_pic.jpg')})

                        leg_select = 62
                        y_label = [51, 61]
                        print('\n'+'#'*50)
                        break
                except:
                    pass
                print('Input error, please try again!!')
                print('\n'+'#'*50)

            # Model - Loading
            train_path = 'EEG Data'
            sub_name = input('Please input the filename of Training data:\n')
            try:
                X_fs_psd = np.load(train_path+'/'+sub_name+'-kneebo_fs_psd_'+leg_name+'.npy')
                clf_psd = joblib.load(train_path+'/'+sub_name+'-kneebo_trained_clf_psd_'+leg_name+'.joblib')

                clf_pcc = joblib.load(train_path+'/'+sub_name+'-kneebo_trained_clf_pcc_'+leg_name+'.joblib')

                csp = joblib.load(train_path+'/'+sub_name+'-kneebo_trained_csp_'+leg_name+'.joblib')
                clf_csp = joblib.load(train_path+'/'+sub_name+'-kneebo_trained_clf_csp_'+leg_name+'.joblib')
                print('\n'+'#'*50)
                break
            except:
                pass
            print('Input error or the model files do not exist, please try again!!')
            print('\n'+'#'*50)

    # Arguments Input & Pre-work
    while True:
        trial_num = input('Please input trial number of each event:\n')
        try:
            trial_num = int(trial_num)
            print('\n'+'#'*50)
            break
        except:
            pass
        print('Input error, please try again!!')
        print('\n'+'#'*50)

    # Training Phase - Motor Detection w/ IMU & Mouse - LSL Inlet
    # subprocess.Popen('IMU - Notch - LSL Marker (PC)/IMU - Notch - LSL Marker.exe')
    subprocess.Popen('./mouse.exe')
    IMU_streams = resolve_stream('name', 'MouseButtons')
    IMU_inlet = StreamInlet(IMU_streams[0])
    IMU_sample_temp, start_lsl_timestp = IMU_inlet.pull_sample()
    print('\n'+'IMU & Mouse LSL Object initilized!!')

    # Testing Phase
    if phase_select == 'Test':
        # Motor Detection w/ EEG - LSL Inlet
        EEG_streams = resolve_stream('type', 'EEG')
        EEG_inlet = StreamInlet(EEG_streams[0])
        EEG_sample_temp, start_lsl_timestp = EEG_inlet.pull_sample()
        print('\nEEG LSL Object initilized!!')

        # Initialize EEG & MNE Information
        plt.pause(0.1)
        EEG_chunk, EEG_chunk_lsl_timestp = EEG_inlet.pull_chunk()
        if len(EEG_chunk[0]) == 8:
            EEG_ch_label = ['Fp1', 'Fp2', 'Fz', 'C3', 'C4', 'Pz', 'O1', 'O2']
            EEG_sprate = 1000
            nfft = 1024
            EEG_bandpass = [7, 30]
            vol_th = 200
            ch_th = 0.5
        elif len(EEG_chunk[0]) == 32:
            EEG_ch_label = ['Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 'FC4', 'FT8', 'T7',
                            'C3', 'Cz', 'C4', 'T8', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']
            EEG_sprate = 500
            nfft = 512
            EEG_bandpass = [0.5, 30]
            vol_th = 50
            ch_th = 0.2

        EEG_mne_info = mne.create_info(ch_names=EEG_ch_label, sfreq=EEG_sprate, ch_types='eeg', verbose=None)
        biosemi_montage = mne.channels.make_standard_montage('standard_1020')
        EEG_mne_info.set_montage(biosemi_montage)

        # Construct an UDP Socket object to connect to the KneeBO Exoskeleton
        kneebo_client_obj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print('\nKneeBO UDP Socket Object initilized!!')

        # Open KneeBO Control Server
        # subprocess.Popen('py ./KneeBO_control_server.py')

    # Event Marker - LSL Outlet
    info = StreamInfo(event_stream_name, 'Markers', 1, 0, 'int32', event_stream_name.replace(' ', '_'))
    event_outlet = StreamOutlet(info)
    print('\nEvent Marker LSL Object initilized!!')
    event_outlet.push_sample([11])

    # Action Cue List - Arrangement
    task_order_arr = np.tile(np.array(list(cue_pic_dict.keys())), trial_num)
    np.random.shuffle(task_order_arr)
    np.random.shuffle(task_order_arr)
    task_order_list = task_order_arr.tolist()

    ################################################################################
    # Experiment Scene - Execution
    # Experiment Scene
    plt.ion()
    exp_fig = plt.figure('BCI KneeBO System (by Lofan)', figsize=(8, 6))
    gs = gridspec.GridSpec(6, 2)
    plt.imshow(ready_pic)
    plt.axis('off')
    exp_fig.canvas.flush_events()
    while True:
        exp_fig.canvas.mpl_connect('button_press_event', onclick)
        if click == True:
            break
        else:
            plt.pause(1)

    # Experiment Countdown
    for init_count_pic in init_count_pic_list:
        plt.cla()
        plt.imshow(init_count_pic)
        plt.axis('off')
        exp_fig.canvas.flush_events()
        plt.pause(1)

    # Experiment Start
    event_outlet.push_sample([11])
    plt.cla()
    plt.subplot(gs[1:, :])
    plt.imshow(start_pic)
    plt.axis('off')
    exp_fig.canvas.flush_events()

    plt.pause(3)
    plt.cla()
    plt.axis('off')

    # Score - Initilization
    plt.subplot(gs[0, 1])
    score = 0
    plt.text(0, 0.35, 'Score: %08d' % (score), fontsize=36, fontweight='light')
    plt.axis('off')
    exp_fig.canvas.flush_events()

    # Tasks - Start
    y_test = []
    y_predict = []
    task_i = 0
    while task_i < len(task_order_list):
        print('Trial No.', task_i+1)

        # Fixation Cue
        plt.subplot(gs[1:, :])
        plt.cla()
        plt.imshow(fixation_pic)
        plt.axis('off')
        exp_fig.canvas.flush_events()
        event_outlet.push_sample([22])
        print('Fixation:', 22)
        plt.pause(1)

        # LSL Streaming - Flush
        IMU_chunk, IMU_lsl_timestp = IMU_inlet.pull_chunk()
        while phase_select == 'Test':
            EEG_chunk, EEG_lsl_timestp = EEG_inlet.pull_chunk()
            EEG_chunk = np.array(EEG_chunk)
            if EEG_chunk.shape[0] < 25:
                break
        EEG_buffer = []
        X_epoch = []

        # Action Cue
        plt.subplot(gs[1:, :])
        plt.cla()
        plt.imshow(cue_pic_dict[task_order_list[task_i]])
        plt.axis('off')
        exp_fig.canvas.flush_events()
        event_outlet.push_sample([task_order_list[task_i]])
        print('Action Cue:', task_order_list[task_i])

        # Trial Countdown - Initialization
        feedback_sample = None
        task_start = time.time()
        task_cnt_i = 0
        task_switch_list = [True] * (len(task_count_pic_list)+1)
        while time.time() - task_start < len(task_count_pic_list):
            # Trial Countdown
            if time.time() - task_start > task_cnt_i and task_switch_list[task_cnt_i] == True:
                plt.subplot(gs[0, 0])
                plt.cla()
                plt.imshow(task_count_pic_list[task_cnt_i])
                plt.axis('off')
                exp_fig.canvas.flush_events()
                task_switch_list[task_cnt_i] = False
                if task_cnt_i < len(task_count_pic_list)-1:
                    task_cnt_i += 1

            # Testing Phase - Motor Detection w/ EEG
            if phase_select == 'Test':
                plt.pause(0.2)
                EEG_chunk, EEG_chunk_lsl_timestp = EEG_inlet.pull_chunk()
                EEG_chunk = np.array(EEG_chunk).T
                try:
                    EEG_buffer = np.hstack([EEG_buffer, EEG_chunk])
                except:
                    EEG_buffer = EEG_chunk
                if EEG_buffer.shape[1] > EEG_sprate * 1.2:
                    X_epoch = EEG_epoch_preprocess(EEG_buffer, EEG_mne_info, EEG_ch_label, sprate=EEG_sprate, bandpass=EEG_bandpass, vol_th=vol_th, ch_th=ch_th)
                    if len(X_epoch) == 0:
                        feedback_sample = 'noisy'
                    else:
                        feedback_sample = EEG_ML_KneeBO(
                            X_epoch[np.newaxis], y_label, X_fs_psd, clf_psd, clf_pcc, csp, clf_csp, sprate=EEG_sprate, nfft=nfft, bandpass=EEG_bandpass)
                        y_test.append(task_order_list[task_i])
                        if feedback_sample == 42:
                            y_predict.append(41)
                        elif feedback_sample == None:
                            y_predict.append(51)
                        elif feedback_sample == 62:
                            y_predict.append(61)

            # Training Phase - Motor Detection w/ IMU & Mouse
            IMU_chunk, IMU_lsl_timestp = IMU_inlet.pull_chunk()
            if len(IMU_chunk) > 0:
                feedback_sample = IMU_chunk[-1][0]
                if feedback_sample[11:15] == 'Left':
                    feedback_sample = 42
                elif feedback_sample[11:16] == 'Right':
                    feedback_sample = 62
                elif feedback_sample[5:10] == 'Wheel':
                    feedback_sample = None
            elif phase_select == 'Train' and time.time() - task_start > 1.5:
                if task_order_list[task_i] == 41:
                    feedback_sample = 42
                elif task_order_list[task_i] == 61:
                    feedback_sample = 62

            # Motor Detection - Figure Feedback
            # No Any Detection
            if phase_select == 'Train' and feedback_sample == None or phase_select == 'Test' and len(X_epoch) == 0 and feedback_sample == None:
                continue

            elif phase_select == 'Test' and feedback_sample == 'noisy':
                event_outlet.push_sample([33])
                print('Trial Feedback:', 33, '(Noisy!!)')
                task_order_list.append(task_order_list[task_i])

                plt.subplot(gs[0, 0])
                plt.cla()
                plt.axis('off')

                plt.subplot(gs[1:, :])
                plt.cla()
                plt.imshow(noisy_pic)
                plt.axis('off')

                exp_fig.canvas.flush_events()
                break
            elif phase_select == 'Test' and len(X_epoch) > 0 and feedback_sample == None:
                break

            # Motor Task - Correct
            elif feedback_sample == task_order_list[task_i]+1:
                event_outlet.push_sample([task_order_list[task_i]+1])
                print('Trial Feedback:', task_order_list[task_i]+1, '(Correct!!)')

                # Figure Feedback
                plt.subplot(gs[0, 0])
                plt.cla()
                plt.axis('off')

                if phase_select == 'Train':
                    score_temp = 100 * (1 - round((time.time() - task_start)/len(task_count_pic_list), 1))
                else:
                    score_temp = 100
                score += score_temp
                plt.subplot(gs[0, 1])
                plt.cla()
                plt.text(0, 0.35, 'Score: %08d' % (score), fontsize=36, fontweight='light')
                plt.axis('off')

                plt.subplot(gs[1:, :])
                plt.cla()
                plt.title('+ %d' % (score_temp), fontsize=18, fontweight='bold')
                plt.imshow(true_pic)
                plt.axis('off')

                exp_fig.canvas.flush_events()

                # KneeBO Feedback
                if phase_select == 'Test' and feedback_sample == leg_select:
                    kneebo_client_obj.sendto('move'.encode(), (kneebo_server_ip, kneebo_server_port))
                    plt.pause(5)
                break

            # Motor/Rest Task - Incorrect (Wrong Action)
            else:
                event_outlet.push_sample([33])
                print('Trial Feedback:', 33, '(Wrong!!)')
                task_order_list.append(task_order_list[task_i])

                plt.subplot(gs[0, 0])
                plt.cla()
                plt.axis('off')

                score -= 100
                plt.subplot(gs[0, 1])
                plt.cla()
                plt.text(0, 0.35, 'Score: %08d' % (score), fontsize=36, fontweight='light')
                plt.axis('off')

                plt.subplot(gs[1:, :])
                plt.cla()
                plt.title('- 100', fontsize=18, fontweight='bold')
                plt.imshow(false_pic)
                plt.axis('off')

                exp_fig.canvas.flush_events()
                break
        # Rest Task - Correct
        if task_order_list[task_i] == 51 and feedback_sample == None:
            event_outlet.push_sample([53])
            print('Trial Feedback:', 53, '(Correct!!)')

            plt.subplot(gs[0, 0])
            plt.cla()
            plt.axis('off')

            score += 100
            plt.subplot(gs[0, 1])
            plt.cla()
            plt.text(0, 0.35, 'Score: %08d' % (score), fontsize=36, fontweight='light')
            plt.axis('off')

            plt.subplot(gs[1:, :])
            plt.cla()
            plt.title('+ 100', fontsize=18, fontweight='bold')
            plt.imshow(true_pic)
            plt.axis('off')

            exp_fig.canvas.flush_events()

        # Motor Task - Incorrect (Late)
        elif feedback_sample == None:
            event_outlet.push_sample([33])
            print('Trial Feedback:', 33, '(Late!!)')
            task_order_list.append(task_order_list[task_i])

            plt.subplot(gs[0, 0])
            plt.cla()
            plt.axis('off')

            score -= 100
            plt.subplot(gs[0, 1])
            plt.cla()
            plt.text(0, 0.35, 'Score: %08d' % (score), fontsize=36, fontweight='light')
            plt.axis('off')

            plt.subplot(gs[1:, :])
            plt.cla()
            plt.title('- 100', fontsize=18, fontweight='bold')
            plt.imshow(late_pic)
            plt.axis('off')

            exp_fig.canvas.flush_events()
        print('\n'+'#'*50)
        plt.pause(1)

        plt.cla()
        plt.axis('off')
        exp_fig.canvas.flush_events()

        plt.pause(1)
        task_i += 1
    plt.imshow(end_pic)
    plt.axis('off')
    exp_fig.canvas.flush_events()
    event_outlet.push_sample([99])
    if phase_select == 'Test':
        kneebo_client_obj.close()
        print('Accuracy:', accuracy_score(y_test, y_predict))
        np.save(train_path+'/'+sub_name+'-kneebo_y_test_'+leg_name+'.npy', y_test)
        np.save(train_path+'/'+sub_name+'-kneebo_y_predict_'+leg_name+'.npy', y_predict)

    print('='*50+'\n\nThe experiment is over, the window will close in 30 seconds!!')
    plt.pause(30)
    plt.ioff()


def onclick(event):
    global click
    if 590 <= event.xdata and event.xdata <= 1900 and 1700 <= event.ydata and event.ydata <= 2000:
        click = True
    else:
        click = False


def IIR_filter(X_epoch=[], fre_cutoff=[0.5, 50], sprate=500, axis=0, pass_type='bandpass', ftype='butter', filter_order=10, filter_plot=False, ch_label=[]):
    # IIR filter - filter design
    sos = signal.iirfilter(filter_order, Wn=fre_cutoff, btype=pass_type, analog=False, ftype=ftype, output='sos', fs=sprate)

    # IIR filter - signal bandpass
    if len(X_epoch) > 0:
        X_epoch_filtered = signal.sosfiltfilt(sos, X_epoch, axis=axis)

    # IIR filter - filter performance plot (optional)
    if filter_plot == True:
        f_axis, h = signal.sosfreqz(sos, 512, fs=sprate)
        plt.figure(figsize=(12, 9))

        if len(X_epoch) > 0 and X_epoch.ndim <= 3:
            plt.subplot(311)
        plt.plot(f_axis, 20 * np.log10(np.maximum(abs(h), 1e-5)))
        try:
            plt.plot([fre_cutoff[0], fre_cutoff[0]], [-105, 5], '--', label=str(fre_cutoff[0])+' Hz')
            plt.plot([fre_cutoff[1], fre_cutoff[1]], [-105, 5], '--', label=str(fre_cutoff[1])+' Hz')
            plt.xticks(np.hstack([np.arange(0, fre_cutoff[1]+20, 5), fre_cutoff[0], fre_cutoff[1]]))
            plt.xlim(0, fre_cutoff[1]+20)
        except:
            plt.plot([fre_cutoff, fre_cutoff], [-105, 5], '--', label=str(fre_cutoff)+' Hz')
            plt.xticks(np.hstack([np.arange(0, fre_cutoff+20, 5), fre_cutoff]))
            plt.xlim(0, fre_cutoff+20)
        plt.yticks(np.arange(-100, 10, 10))
        plt.ylim(-100, 10)
        plt.title('IIR ('+ftype+') Filter frequency response')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Quantity (dB)')
        plt.legend(loc='lower right')
        plt.grid()

        if len(X_epoch) > 0 and X_epoch.ndim <= 3:
            if len(ch_label) == 0 and X_epoch.ndim >= 2:
                ch_label = ['Ch'+str(ch) for ch in range(X_epoch.shape[1])]

            t_axis = np.arange(0, X_epoch.shape[axis]/sprate, 1/sprate)
            if X_epoch.ndim == 3:
                X_epoch = X_epoch.mean(axis=0).T
                X_epoch_filtered_plot = X_epoch_filtered.mean(axis=0).T
            else:
                X_epoch_filtered_plot = X_epoch_filtered

            plt.subplot(312)
            plt.plot(t_axis, X_epoch, label=ch_label)
            plt.title('before filtering')
            plt.xlabel('Time (sec)')
            plt.ylabel('Voltage (μV)')
            plt.xlim(t_axis.min(), t_axis.max())
            if len(ch_label) <= 10:
                plt.legend(loc='lower right')
            plt.grid()

            plt.subplot(313)
            plt.plot(t_axis, X_epoch_filtered_plot, label=ch_label)
            plt.title('after filtering')
            plt.xlabel('Time (sec)')
            plt.ylabel('Voltage (μV)')
            plt.xlim(t_axis.min(), t_axis.max())
            if len(ch_label) <= 10:
                plt.legend(loc='lower right')
            plt.grid()
        plt.tight_layout()
        plt.show()

    if len(X_epoch) > 0:
        return X_epoch_filtered


def EEG_epoch_preprocess(EEG_epoch, EEG_mne_info, EEG_ch_label, sprate=500, bandpass=[0.5, 30], vol_th=50, ch_th=0.2):
    # Bandpass Filter
    EEG_epoch_ini = IIR_filter(EEG_epoch[:, :int(sprate*1.2)], fre_cutoff=bandpass, sprate=sprate, axis=1)
    EEG_epoch_filtered = EEG_epoch_ini[:, int(sprate*0.5):int(sprate*1)]
    print('Bandpass', EEG_epoch_filtered.shape)

    # Detect Bad Trial
    bad_epoch_ch, bad_epoch_sp = np.where((EEG_epoch_filtered < -vol_th) | (vol_th < EEG_epoch_filtered))
    if len(np.unique(bad_epoch_ch)) > EEG_epoch.shape[0] * ch_th:
        return []

    # Interpolate Bad Channels - 1
    EEG_mne_info['bads'] = []
    for ch_i in range(EEG_epoch_filtered.shape[0]):
        ch_max = np.abs(EEG_epoch_filtered[ch_i, :]).max()
        if ch_max > vol_th and EEG_ch_label[ch_i] not in EEG_mne_info['bads']:
            print(EEG_ch_label[ch_i], '- amp:', ch_max)
            EEG_mne_info['bads'].append(EEG_ch_label[ch_i])
    if len(EEG_mne_info['bads']) > 0:
        EEG_epoch_mne = mne.EvokedArray(EEG_epoch_filtered, EEG_mne_info)
        print(EEG_mne_info['bads'])
        EEG_epoch_mne.interpolate_bads()
        EEG_epoch_filtered = EEG_epoch_mne.data
    EEG_bad_ch = EEG_mne_info['bads']

    # ## Interpolate Bad Channels - 2
    # EEG_mne_info['bads'] = []
    # for ch_i in range(EEG_epoch_filtered.shape[0]):
    #     ch_corr = np.corrcoef(np.vstack([EEG_epoch_filtered[ch_i,:], EEG_epoch_filtered.mean(axis=0)]))
    #     ch_std = np.var(EEG_epoch_filtered[ch_i,:])/np.var(EEG_epoch_filtered)
    #     if ch_corr[0,1] <= 0 or ch_std >= 3 and EEG_ch_label[ch_i] not in EEG_mne_info['bads']:
    #         if ch_std >= 3:
    #             print(EEG_ch_label[ch_i],'- std:',ch_std)
    #         elif ch_corr[0,1] <= 0:
    #             print(EEG_ch_label[ch_i],'- corr:',ch_corr[0,1])
    #         EEG_mne_info['bads'].append(EEG_ch_label[ch_i])
    # if len(set(EEG_bad_ch+EEG_mne_info['bads'])) > EEG_epoch.shape[0] * 0.5:
    #     return []
    # if len(EEG_mne_info['bads']) > 0:
    #     EEG_epoch_mne = mne.EvokedArray(EEG_epoch_filtered, EEG_mne_info)
    #     print(EEG_mne_info['bads'])
    #     EEG_epoch_mne.interpolate_bads()
    #     EEG_epoch_filtered = EEG_epoch_mne.data

    return EEG_epoch_filtered


def ML_X_corr(X_epoch):
    X_corr = []

    for X_epoch_single in X_epoch:
        EEG_corr = np.corrcoef(X_epoch_single)
        EEG_corr_triu_i = np.triu_indices_from(EEG_corr, k=1)

        X_corr.append(EEG_corr[EEG_corr_triu_i])
    X_corr = np.vstack(X_corr)

    return X_corr

# 在測試階段用，用於判斷當前的 EEG 訊號對應的輸出
def EEG_ML_KneeBO(X_epoch, y_label, X_fs_psd, clf_psd, clf_pcc, csp, clf_csp, sprate=500, nfft=512, bandpass=[0.5, 30]):
    # 1. 功率譜密度 PSD(Power Spectral Density)
    f_axis_welch, X_welch = signal.welch(X_epoch, sprate, window='hamming', nperseg=sprate//2, noverlap=sprate//4, nfft=nfft, axis=2)
    f_axis_welch_i, = np.where((bandpass[0] <= f_axis_welch) & (f_axis_welch <= bandpass[1]))
    X_psd = X_welch[:, :, f_axis_welch_i].reshape(X_welch.shape[0], X_welch.shape[1]*f_axis_welch_i.shape[0])
    y_proba_psd = clf_psd.predict_proba(X_psd[:, X_fs_psd])

    # 2. 皮爾森相關係數 PCC (Pearson Correlation Coefficient)
    X_pcc = ML_X_corr(X_epoch)
    y_proba_pcc = clf_pcc.predict_proba(X_pcc)

    # 3. 共通空間模式 CSP (Common Spatial Pattern)
    X_csp = csp.transform(X_epoch)
    y_proba_csp = clf_csp.predict_proba(X_csp)

    # Mix
    y_proba = (y_proba_psd + y_proba_pcc + y_proba_csp).squeeze()
    y_predict = y_label[y_proba.argmax()]

    if y_predict == 41:
        return 42
    elif y_predict == 61:
        return 62
    elif y_predict == 51:
        return None


if __name__ == '__main__':
    BCI_KneeBO()
