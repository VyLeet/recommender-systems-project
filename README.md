# Recommender systems project
Repository for Recommender Systems UCU course project

## Report
The final report and additions are in the file [report.html](report.html).

The [report.ipynb](report.ipynb) is the same file in the jupyter notebook format. 

## Quick Start

### Clone the repository and set up the environment 
1. Clone the repository with `git clone`
2. Create a virtual environment with `virtualenv venv`
3. Activate the virtual environment with `source venv/bin/activate`
4. Install the dependencies with `pip install -r requirements.txt`

### Run the Evaluation Framework CLI:
1. Train, evaluate and save a model to a file: `python train.py -m content_based --content_based.tfidf_max_features 40 --evaluate --save /tmp/model.pkl` 
2. Evaluate a previously saved model: `python evaluate.py -m /tmp/model.pkl` 
3. Run inference and save results to a CSV file: `python inference.py -m /tmp/model.pkl -o /tmp/results.csv` 

## Collaborators
<table>
  <tr>
    <td align="center">
      <img src="https://media.licdn.com/dms/image/D4D03AQFP6HBqv2wwfg/profile-displayphoto-shrink_800_800/0/1672130410834?e=1691625600&v=beta&t=3Yxkabb0Tm6wj_hj2XG-cAUessVr0dRAuphxzZTiLOg" alt="Nazariy Vysokinskyi" width="100" style="border-radius: 50%;">
      <p>Nazariy Vysokinskyi</p>
    </td>
    <td align="center">
      <img src="https://cdn.iconscout.com/icon/free/png-256/free-avatar-370-456322.png?f=webp" alt="Dmytro Ponomarov" width="100" style="border-radius: 50%;">
      <p>Dmytro Ponomarov</p>
    </td>
    <td align="center">
      <img src="https://media.licdn.com/dms/image/C5603AQGXQqM0arUrXA/profile-displayphoto-shrink_800_800/0/1516432360820?e=1695859200&v=beta&t=6RT0PJ3bm2rXkuwFA-26_Kgn6lv0ZELM1d5YELf_kWI" alt="Maksym Sarana" width="100" style="border-radius: 50%;">
      <p>Maksym Sarana</p>
    </td>
  </tr>
</table>
