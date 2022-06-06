# Taipower-Electricity-Forecasting

<!-- PROJECT LOGO -->
<br />
<p align="center">

  <h3 align="center">Taipower Electricity Forecasting</h3>

  <p align="center">
    DSAI HW1
    <br />
    <a href="https://github.com/Shulammiteya/Taipower-Electricity-Forecasting"><strong>Explore the docs »</strong></a>
    <br />
  </p>
</p>


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* Python 3.7.13

### Usage

1. Install packages
   ```sh
   pip install -r requirements.txt
   ```
2. Run app.py and don't train （The parameter 'train' identifies whether to retrain the model）
   ```JS
   python app.py --training training_data.csv --output submission.csv --train False
   ```
3. Run app.py and train
   ```JS
   python app.py --training training_data.csv --output submission.csv --train True
   ```
<br />


## data analysis

* y軸為：operating reserve (備轉容量)，資料無缺失值。
<p float="center" align="center">
<img src="https://drive.google.com/uc?export=view&id=1_HR6qqQChNkNMXSn3PPzopNedHhUbdIa" alt="data analysis">
</p>


## data scaling

* 使用 sklearn 的 MinMaxScaler
   ```JS
   
   scaler = MinMaxScaler(feature_range=(0, 1))

  def normalize(data):
    norm_arr = scaler.fit_transform(data)
    return pd.DataFrame(norm_arr)

  def inverse_normalize(data):
    return scaler.inverse_transform(data)
    
   ```


## model training

* 使用 LSTM 模型進行多步預測，輸入為前 30 天的電力資訊，輸出為未來 15 天的備轉容量預測。
<p float="center" align="center">
<img src="https://drive.google.com/uc?export=view&id=1qaUWQuPJkO_HhzViuC9uPhTJ3HToDejm" alt="model structure">
</p>

* 模型訓練資訊。
<p float="center" align="center">
<img src="https://drive.google.com/uc?export=view&id=1ZxzO1JfvDsTrW9aY-KCAM-iNAVVjNDHA" alt="training history">
</p>
<br />



<!-- CONTACT -->
## Contact

About me: [Hsin-Hsin, Chen](https://www.facebook.com/profile.php?id=100004017297228) - shulammite302332@gmail.com
