import numpy as np
np.set_printoptions(suppress=True)
from scipy import io, signal, stats
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectKBest, f_classif, SequentialFeatureSelector
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit, KFold, LeaveOneOut, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

import mne
from mne.decoding import CSP
from mrmr import mrmr_classif

import os
import joblib
from tkinter import filedialog

import sys
os.chdir(sys.path[0])#將當前環境位置設為當前"檔案位置"

# ==============================================================================
# global 
EEG_event_list = [411,421,51,611,621]
failed_marker = 33
left_event = [411,421]
right_event = [611,621]
# ==============================================================================
def hand_selection():
  """選擇 左臂/右臂"""
  while True:
    hand_select = input('Please select the side of hand:\n  1) Left hand\n  2) Right hand\n')
    if hand_select == '1':
      return ('l',right_event)
    elif hand_select == '2':
      return ('r',left_event)
    print('Input error, please try again!!')
    print('\n'+'#'*50)

def get_EEG_info(EEG_ch_label):
  EEG_info = {}
  if EEG_ch_label[10] == 'Event Id':
    EEG_info['state'] = True
    EEG_info['EEG_ch_num'] = 8
    EEG_info['EEG_sprate'] = 1000
    EEG_info['nfft'] = 1024
    EEG_info['EEG_bandpass'] = [7,30]
    EEG_info['vol_th'] = 200
    EEG_info['ch_th'] = 0.5
    return EEG_info
  elif EEG_ch_label[34] == 'Event Id':
    EEG_info['state'] = True
    EEG_info['EEG_ch_num'] = 32
    EEG_info['EEG_sprate'] = 500
    EEG_info['nfft'] = 512
    EEG_info['EEG_bandpass'] = [0.5,30]
    EEG_info['vol_th'] = 50
    EEG_info['ch_th'] = 0.2
    return EEG_info
  return None

def IIR_filter(X_epoch=[], fre_cutoff=[0.5, 50], sprate=500, axis=0, pass_type='bandpass', ftype='butter', filter_order=30):
  # IIR filter - filter design
  sos = signal.iirfilter(filter_order, Wn=fre_cutoff, btype=pass_type, analog=False, ftype=ftype, output='sos', fs=sprate)
  # IIR filter - signal bandpass
  if len(X_epoch) > 0:
    X_epoch_filtered = signal.sosfiltfilt(sos, X_epoch, axis=axis)
    return X_epoch_filtered

class EEGLAB(object):
  def __init__(self,EEG_file_path,eeg_info):
    self.EEG_ch_num = eeg_info['EEG_ch_num']
    self.EEG_sprate = eeg_info['EEG_sprate']
    self.nfft = eeg_info['nfft']
    self.EEG_bandpass = eeg_info['EEG_bandpass']
    self.vol_th = eeg_info['vol_th']
    self.ch_th = eeg_info['ch_th']
    self.EEG_file_path = EEG_file_path
    self.marker_positions = {}# 紀錄每個event marker的位置
    # 取樣範圍
    self.X_epoch_sp_start_i = int(self.EEG_sprate * 0.5)
    self.X_epoch_sp_end_i = int(self.EEG_sprate * 1)
  # Data Preprocessing
  def data_preprocessing(self,show_msg=True):
    ## Load EEG csv File
    data_range = [np.arange(2,self.EEG_ch_num+2),self.EEG_ch_num+5]
    EEG_data_raw = np.loadtxt(self.EEG_file_path, dtype=str, skiprows=10, usecols=np.hstack(data_range), delimiter=',', encoding='utf-8')

    self.EEG_ch_label = EEG_data_raw[0,:-1]
    self.EEG_data = EEG_data_raw[1:,:-1].astype(float).T
    self.EEG_event_data = np.where(EEG_data_raw[1:,-1] == '', 0, EEG_data_raw[1:,-1]).astype(float)
    if show_msg:
      print('EEG channels:', self.EEG_ch_label.shape, EEG_ch_label, sep='\n', end='\n\n')
      print('EEG data:', self.EEG_data.shape, self.EEG_data[:5,:], sep='\n', end='\n\n')
      print('EEG event markers:', self.EEG_event_data.shape, np.unique(self.EEG_event_data), sep='\n', end='\n\n')
  # 紀錄每個event marker的位置
  def record_marker_positions(self):
    elements = sorted(list(set(self.EEG_event_data)))
    for EEG_event in elements:
      if EEG_event!=0:
        EEG_event = int(EEG_event)
        EEG_event_i_vec, = np.where(self.EEG_event_data == EEG_event)
        self.marker_positions[EEG_event] = EEG_event_i_vec
  # Initialize MNE Information
  def init_MNE(self):
    self.EEG_mne_info = mne.create_info(ch_names=self.EEG_ch_label.tolist(), sfreq=self.EEG_sprate, ch_types='eeg', verbose=None)
    biosemi_montage = mne.channels.make_standard_montage('standard_1020')
    self.EEG_mne_info.set_montage(biosemi_montage)
  # Failed Trials Detection(失敗檢測)__Extract Epochs時用
  def failed_trials_detection(self,marker_positions,EEG_event_data,EEG_event_i,EEG_sprate):
    if failed_marker in marker_positions:
      block = EEG_event_data[EEG_event_i:EEG_sprate*2]
      if failed_marker not in block:
        return True
    return False  
  # Extract Epochs postprocessing
  def EEG_epoch_postprocessing(self,EEG_epoch_filtered):
    ## Remove Bad Trials() 檢查 epoch是否存在幅度超出門檻值的情況，如果異常通道的數量超過設定的ch_th比例，直接跳過該 epoch。
    bad_epoch_ch,bad_epoch_sp = np.where((EEG_epoch_filtered < -self.vol_th) | (self.vol_th < EEG_epoch_filtered))
    if len(np.unique(bad_epoch_ch)) > self.EEG_ch_num * self.ch_th:
      return None
    ## Interpolate Bad Channels - 1  #檢查 epoch是否存在幅度超出門檻值的情況，有異常時插值，跟前面那段有點重複，先不刪
    self.EEG_mne_info['bads'] = []
    for ch_i in range(EEG_epoch_filtered.shape[0]):
      ch_max = np.abs(EEG_epoch_filtered[ch_i,:]).max()
      if ch_max > self.vol_th and self.EEG_ch_label[ch_i] not in self.EEG_mne_info['bads']:
        print(self.EEG_ch_label[ch_i],'- amp:',ch_max)                    
        self.EEG_mne_info['bads'].append(self.EEG_ch_label[ch_i])
    if len(self.EEG_mne_info['bads']) > 0:
      EEG_epoch_mne = mne.EvokedArray(EEG_epoch_filtered, self.EEG_mne_info)
      print(self.EEG_mne_info['bads'])
      EEG_epoch_mne.interpolate_bads()
      EEG_epoch_filtered = EEG_epoch_mne.data
    EEG_bad_ch = self.EEG_mne_info['bads']
    return EEG_epoch_filtered
  # Extract Epochs
  def extract_epochs(self):
    EEG_epoch = []
    EEG_label = []
    for EEG_event in EEG_event_list:
      EEG_event_i_vec, = np.where(self.EEG_event_data == EEG_event)
      for EEG_event_i in EEG_event_i_vec:
        ## 失敗檢測
        break_power = self.failed_trials_detection(self.marker_positions,self.EEG_event_data,EEG_event_i,self.EEG_sprate)
        if break_power == True:
          continue
        ## Bandpass Filter(帶通濾波)
        if EEG_event_i + self.EEG_sprate*1.2 > self.EEG_data.shape[1]:
          continue # 超出範圍了
        EEG_data_epoch = self.EEG_data[:,EEG_event_i:int(EEG_event_i+self.EEG_sprate*1.2)]
        EEG_epoch_filtered_ini = IIR_filter(EEG_data_epoch, 
                                            fre_cutoff=self.EEG_bandpass, 
                                            sprate=self.EEG_sprate, 
                                            axis=1)
        EEG_epoch_filtered = EEG_epoch_filtered_ini[:,self.X_epoch_sp_start_i:self.X_epoch_sp_end_i]
        EEG_epoch_filtered = self.EEG_epoch_postprocessing(EEG_epoch_filtered)
        if EEG_epoch_filtered is None:
          continue
        EEG_epoch.append(EEG_epoch_filtered)
        EEG_label.append(EEG_event)
    # list -> numpy_array
    EEG_epoch = np.array(EEG_epoch)
    EEG_label = np.array(EEG_label)
    
    return EEG_epoch,EEG_label
  ## 可視化預處理的EEG Data
  def visualize_preprocessed_EEG_data(self,EEG_epoch,EEG_visu_path):
    for ch_8_i in range(len(self.EEG_ch_label)//8):
      plt.figure(figsize=(12,3))
      plt.plot(EEG_epoch.mean(axis=0)[8*ch_8_i:8*(ch_8_i+1),:].T, label=self.EEG_ch_label[8*ch_8_i:8*(ch_8_i+1)])
      plt.yticks(np.arange(-10,11,2))
      plt.xlim([0,(self.X_epoch_sp_end_i-self.X_epoch_sp_start_i)])
      plt.ylim([-10,10])
      plt.ylabel('Voltage (μV)')
      plt.legend(loc='lower right')
      plt.savefig(f'{EEG_visu_path}{ch_8_i}.png')
      
      plt.close()
# 相關係數矩陣
def ML_X_corr(X_epoch):
  X_corr = []
  for X_epoch_single in X_epoch:
    EEG_corr = np.corrcoef(X_epoch_single)
    EEG_corr_triu_i = np.triu_indices_from(EEG_corr, k=1)
    
    X_corr.append(EEG_corr[EEG_corr_triu_i])
  X_corr = np.vstack(X_corr)
  return X_corr 

class MachineLearning(object):
  def __init__(self,EEG_epoch,EEG_label):
    self.lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    self.qda = make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis())
    self.lin_svm = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True, random_state=2023))
    self.svm = make_pipeline(StandardScaler(), SVC(probability=True, random_state=2023))
    self.knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_jobs=-1))
    self.soft_vote = VotingClassifier(estimators=[('LDA', self.lda), ('QDA', self.qda), ('L-SVM', self.lin_svm), ('SVM', self.svm), ('kNN', self.knn)], 
                                      voting='soft', n_jobs=-1)
    self.n_feature_max = 100
    self.CV_n_splits = 5      # k折交叉驗證折數
    self.cv = KFold(n_splits=self.CV_n_splits, shuffle=True, random_state=2023)
    
    self.EEG_epoch = EEG_epoch
    self.EEG_label = EEG_label
    self.show_result = ''
  # 設定欲排除資料
  def set_exclusion_criteria(self,non_topic_marker):
    sel_list = np.array([i for i,label in enumerate(self.EEG_label) if label not in non_topic_marker])
    self.X_epoch = self.EEG_epoch[sel_list]
    self.y = self.EEG_label[sel_list]
  # k折分割資料
  def split_training_data(self):
    self.cv_split = list(self.cv.split(self.X_epoch))
    self.y_test_all = np.zeros(len(self.y))
    test_n = 0
    for _, test_i in self.cv_split:
      n = len(test_i) # 測試集資料數
      self.y_test_all[test_n:test_n+n] = self.y[test_i]
      test_n += n
  # PSD(功率密度頻譜):
  def PSD_fit(self,eeg_info,fs_psd_path,clf_psd_path):
    clf = self.soft_vote

    EEG_sprate = eeg_info['EEG_sprate']
    nfft = eeg_info['nfft']
    EEG_bandpass = eeg_info['EEG_bandpass']
    f_axis_welch, X_welch = signal.welch(self.X_epoch, EEG_sprate, window='hamming', nperseg=EEG_sprate//2, noverlap=EEG_sprate//4, nfft=nfft, axis=2)
    f_axis_welch_i, = np.where((EEG_bandpass[0] <= f_axis_welch) & (f_axis_welch <= EEG_bandpass[1]))
    X_psd = X_welch[:,:,f_axis_welch_i].reshape(X_welch.shape[0],X_welch.shape[1]*f_axis_welch_i.shape[0])

    X = X_psd
    y = self.y
    N = len(y) # 資料數
    m = len(np.unique(y)) # 標籤數

    y_test_predict = np.zeros((min(X.shape[1], self.n_feature_max), N))
    y_test_proba = np.zeros((min(X.shape[1], self.n_feature_max), N, m))
    y_test = self.y_test_all
    y_test_acc = np.zeros((np.minimum(X.shape[1], self.n_feature_max)))

    test_n = 0
    for train_i, test_i in self.cv_split:
      n = len(test_i) # 測試集資料數
      try:
        X_fs_max_i = mrmr_classif(X=pd.DataFrame(X[train_i]), y=pd.Series(y[train_i]), K=min(X.shape[1], self.n_feature_max))
      except:
        X_fs_max_i = np.arange(X.shape[1])
      for n_feature in range(1, min(X.shape[1], self.n_feature_max)+1):
        X_fs = X[:,X_fs_max_i[:n_feature]]

        clf.fit(X_fs[train_i], y[train_i])
        y_test_predict[n_feature-1,test_n:test_n+n] = clf.predict(X_fs[test_i])
        y_test_proba[n_feature-1,test_n:test_n+n,:] = clf.predict_proba(X_fs[test_i])
      
      test_n += n

    for n_feature in range(1, min(X.shape[1], self.n_feature_max)+1):
      y_test_acc[n_feature-1] = accuracy_score(y_test, y_test_predict[n_feature-1,:])
    
    self.show_result += f'PSD FS number: {np.argmax(y_test_acc)+1} \n'
    self.show_result += f'PSD Accuracy: {np.max(y_test_acc)*100:.2f}% \n'
    self.y_test_proba_psd = y_test_proba[np.argmax(y_test_acc),:,:]
    
    X_fs_i = mrmr_classif(X=pd.DataFrame(X), y=pd.Series(y), K=(np.argmax(y_test_acc)+1))
    clf_psd = clf.fit(X[:,X_fs_i],y)

    np.save(fs_psd_path, X_fs_i)
    joblib.dump(clf_psd, clf_psd_path)
  
  # PCC(皮爾森相關係數):
  def PCC_fit(self,clf_pcc_path):
    clf = self.lin_svm
    y = self.y
    N = len(y) # 資料數
    m = len(np.unique(y)) # 標籤數
    
    y_test_predict = np.zeros(N)
    y_test_proba = np.zeros((N, m))
    y_test = self.y_test_all
    X_pcc = ML_X_corr(self.X_epoch)
    X = X_pcc
    test_n = 0
    for train_i, test_i in self.cv_split:
      n = len(test_i) # 測試集資料數
      clf.fit(X[train_i], y[train_i])
      y_test_predict[test_n:test_n+n] = clf.predict(X[test_i])
      y_test_proba[test_n:test_n+n,:] = clf.predict_proba(X[test_i])
      
      test_n += n
    ## 紀錄準確率
    y_test_acc = accuracy_score(y_test, y_test_predict)
    self.show_result += f'PCC Accuracy:{y_test_acc*100:.2f}%\n'
    self.y_test_proba_pcc = y_test_proba
    ## 用所有資料完整訓練一次
    clf_pcc = clf.fit(X,y)
    joblib.dump(clf_pcc, clf_pcc_path)
  
  # 共通空間模式
  def CSP_fit(self,EEG_ch_num,EEG_mne_info,csp_path,clf_csp_path,img_csp_path):
    clf = self.lda
    y = self.y
    N = len(y) # 資料數
    m = len(np.unique(y)) # 標籤數
    half_ch_num = EEG_ch_num//2

    y_test_predict = np.zeros((half_ch_num, N))
    y_test_proba = np.zeros((half_ch_num, len(y), m))
    y_test = self.y_test_all
    y_test_acc = np.zeros((half_ch_num))

    test_n = 0
    for train_i, test_i in self.cv_split:
      n = len(test_i) # 測試集資料數
      csp = CSP(n_components=EEG_ch_num,reg=0.1)
      csp.fit(self.X_epoch[train_i], y[train_i])
      X_csp = csp.transform(self.X_epoch)
      X = X_csp
      for n_feature in range(2, min(X.shape[1], self.n_feature_max)+1, 2):
        X_fs = X[:,:n_feature]

        clf.fit(X_fs[train_i], y[train_i])
        y_test_predict[n_feature//2-1,test_n:test_n+n] = clf.predict(X_fs[test_i])
        y_test_proba[n_feature//2-1,test_n:test_n+n,:] = clf.predict_proba(X_fs[test_i])
      
      test_n += n

    for n_feature in range(2, min(X.shape[1], self.n_feature_max)+1, 2):
      y_test_acc[n_feature//2-1] = accuracy_score(y_test, y_test_predict[n_feature//2-1,:])

    self.show_result += f'CSP FS number:{(np.argmax(y_test_acc)+1)*2}\n'
    self.show_result += f'CSP Accuracy:{np.max(y_test_acc)*100:.2f}%\n'
    self.y_test_proba_csp = y_test_proba[np.argmax(y_test_acc),:,:]

    csp = CSP(n_components=int((np.argmax(y_test_acc)+1)*2))
    X_csp = csp.fit_transform(self.X_epoch, y)
    clf_csp = clf.fit(X_csp[:,:(np.argmax(y_test_acc)+1)*2], y)

    joblib.dump(csp, csp_path)
    joblib.dump(clf_csp, clf_csp_path)

    ##################################################
    csp.set_params(n_components=8).plot_patterns(EEG_mne_info, cmap='jet', show=False)
    plt.savefig(img_csp_path)
    plt.close()
    
  # mix
  def print_mix_result(self,trials_num):
    y_test_proba_final = self.y_test_proba_psd + self.y_test_proba_pcc + self.y_test_proba_csp
    y_test_predict_temp = y_test_proba_final.argmax(axis=1)
    y = self.y
    m = len(np.unique(y))
    for y_i in range(m):
      y_test_predict_temp = np.where(y_test_predict_temp == y_i, np.unique(y)[y_i], y_test_predict_temp)
            
    y_test_predict_final = y_test_predict_temp
    y_test = self.y_test_all
    print('='*20)
    print(f'資料試驗次數:{trials_num}')
    print(self.show_result)
    print(f'Final Accuracy:{accuracy_score(y_test, y_test_predict_final)*100:.2f}%')
    print(f'Final Precision:{precision_score(y_test, y_test_predict_final, average="weighted")*100:.2f}')
    print(f'Final Recall:{recall_score(y_test, y_test_predict_final, average="weighted")*100:.2f}')
    print(f'Final F1 Score:{f1_score(y_test, y_test_predict_final, average="weighted")*100:.2f}')
  
  
if __name__ == '__main__':
  ## 選擇 左手/右手
  leg_name,non_topic_marker = hand_selection()
  ## 輸入資料集
  try:
    EEG_file_path = filedialog.askopenfilename(title="選擇資料集",filetypes=[("csv files", "*.csv")])
    EEG_file_name = EEG_file_path.split('/')[-1].replace('.csv','')
    EEG_ch_label = np.loadtxt(EEG_file_path, dtype=str, skiprows=10, delimiter=',', encoding='utf-8', max_rows=1)
  except Exception as e:
    input(e)
    exit()
  ## 取得腦波帽相關設定
  eeg_info = get_EEG_info(EEG_ch_label)
  if eeg_info is None:
    input('label 異常')
    exit()
  print('='*20)
  ## 輸出檔案位置設定
  EEG_model_path = 'EEG Data/model'
  EEG_visu_path = f'{EEG_model_path}/{EEG_file_name}-kneebo_EEG_epoch_'
  fs_psd_path   = f'{EEG_model_path}/{EEG_file_name}-kneebo_fs_psd_{leg_name}.npy' 
  clf_psd_path  = f'{EEG_model_path}/{EEG_file_name}-kneebo_trained_clf_psd_{leg_name}.joblib'
  clf_pcc_path  = f'{EEG_model_path}/{EEG_file_name}-kneebo_trained_clf_pcc_{leg_name}.joblib'
  csp_path      = f'{EEG_model_path}/{EEG_file_name}-kneebo_trained_csp_{leg_name}.joblib'
  clf_csp_path  = f'{EEG_model_path}/{EEG_file_name}-kneebo_trained_clf_csp_{leg_name}.joblib'
  img_csp_path  = f'{EEG_model_path}/{EEG_file_name}-kneebo_trained_csp_{leg_name}.png'
  ############################################################################################
  ## 腦波資料處裡:
  EEG = EEGLAB(EEG_file_path,eeg_info)
  EEG.data_preprocessing()
  EEG.record_marker_positions() # 紀錄每個event marker的位置
  EEG.init_MNE() # 用來處理異常腦波對其做插值處理
  EEG_epoch,EEG_label = EEG.extract_epochs() # 提取event data
  print(EEG_label)
  # 計算試驗次數
  trials_num = len(EEG.marker_positions[22])//len(EEG_event_list)
  EEG.visualize_preprocessed_EEG_data(EEG_epoch,EEG_visu_path)
  print('='*20)
  ## Machine Learning
  ML = MachineLearning(EEG_epoch,EEG_label)
  ML.set_exclusion_criteria(non_topic_marker)
  ML.split_training_data()
  ## PSD
  ML.PSD_fit(eeg_info,fs_psd_path,clf_psd_path)
  ## PCC
  ML.PCC_fit(clf_pcc_path)
  ## CSP
  ML.CSP_fit(EEG.EEG_ch_num,EEG.EEG_mne_info,csp_path,clf_csp_path,img_csp_path)
  ML.print_mix_result(trials_num)
  input()
  