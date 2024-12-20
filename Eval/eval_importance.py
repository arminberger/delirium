logistic = """
41                       6:_anaesthesia_Spinal    0.096365
23                  27:_laryngeal_mask_BI_Tube    0.051890
39          6:_anaesthesia_General anaesthesia    0.044159
37       6:_anaesthesia_Cont. Temp. Monitoring    0.038843
38     6:_anaesthesia_EEG used (is protective)    0.037383
27              4:_surgical_speciality_General    0.029591
28              4:_surgical_speciality_Gyn/Obs    0.028932
15                21: patient_uses_dentures_BI    0.027128
29                4:_surgical_speciality_Neuro    0.025959
48               7:_prevention_PONV_(SOP used)    0.025409
40                     6:_anaesthesia_Sedation    0.024238
52                      9:_pain_BI_Resting_NRS    0.023091
20            25:_noise_reduction_performed_BI    0.022268
36  6:_anaesthesia_Cont. Analgesic (painkiller    0.021571
1                       (5:_setting)_Inpatient    0.021318
32         4:_surgical_speciality_Trauma/Orth.    0.020703
8          16:_general_well-being_nrs _0-10_BI    0.020051
51                 9:_pain_BI_Mobilization_NRS    0.019871
7                             15_stress_nrs_BI    0.018598
11             19:_patient_uses_visual_aids_BI    0.018528
"""

xgb = """
41                              6:_anaesthesia_Spinal    0.202384
39                 6:_anaesthesia_General anaesthesia    0.199401
23                         27:_laryngeal_mask_BI_Tube    0.077824
38            6:_anaesthesia_EEG used (is protective)    0.071108
42  6:_anaesthesia_TCI (Target Controlled Infusion...    0.043744
28                     4:_surgical_speciality_Gyn/Obs    0.029249
20                   25:_noise_reduction_performed_BI    0.026182
7                                    15_stress_nrs_BI    0.025416
18                   23:_thirst_feeling: _nrs_0-10_BI    0.022839
26                                          3:_asa-ps    0.022767
37              6:_anaesthesia_Cont. Temp. Monitoring    0.021669
8                 16:_general_well-being_nrs _0-10_BI    0.019530
21                          27:_laryngeal_mask_BI_LMA    0.019354
54                                                age    0.017772
19                              24:_temperature _c_BI    0.017243
17                         22:_fluids_p.o._>200_ml_BI    0.017143
11                    19:_patient_uses_visual_aids_BI    0.016988
43  6:_anaesthesia_TIVA (Total intravenous anesthe...    0.016898
6                                   14_anxiety_nrs_BI    0.014119
15                       21: patient_uses_dentures_BI    0.013665
"""

svm = """
41                              6:_anaesthesia_Spinal    0.091569
39                 6:_anaesthesia_General anaesthesia    0.075116
23                         27:_laryngeal_mask_BI_Tube    0.073451
38            6:_anaesthesia_EEG used (is protective)    0.060757
42  6:_anaesthesia_TCI (Target Controlled Infusion...    0.035197
28                     4:_surgical_speciality_Gyn/Obs    0.032779
36         6:_anaesthesia_Cont. Analgesic (painkiller    0.030618
33                     4:_surgical_speciality_Urology    0.026770
29                       4:_surgical_speciality_Neuro    0.025706
21                          27:_laryngeal_mask_BI_LMA    0.020886
15                       21: patient_uses_dentures_BI    0.019966
27                     4:_surgical_speciality_General    0.019822
24           27:_laryngeal_mask_BI_Video laryngoscope    0.019632
4                                   11:_rass_score_BI    0.019558
37              6:_anaesthesia_Cont. Temp. Monitoring    0.018556
11                    19:_patient_uses_visual_aids_BI    0.018383
50  7:_prevention_Preemptive_Analgesia_(SOP used) ...    0.017124
20                   25:_noise_reduction_performed_BI    0.017063
8                 16:_general_well-being_nrs _0-10_BI    0.016946
40                            6:_anaesthesia_Sedation    0.016935
"""

importances = [logistic, xgb, svm]

# Parse the data
import pandas as pd
import numpy as np
import re


def parse_importance(importance):
    importance = importance.strip()
    importance = importance.split("\n")
    importance = [i.split("    ") for i in importance]
    importance = [i[len(i) - 2 :] for i in importance]
    # Strip the feature names
    importance = [[i[0].strip(), i[1]] for i in importance]
    importance = pd.DataFrame(importance, columns=["feature", "importance"])
    importance["importance"] = importance["importance"].astype(float)
    return importance


importances = [parse_importance(i) for i in importances]

# Get all features that are important in all models and take their mean rank as the final rank
# normalize importance to sum to 1
importances_normalized = []
for importance in importances:
    importance["importance"] = importance["importance"] / importance["importance"].sum()
    importances_normalized.append(importance)
overall_importance = pd.concat(importances_normalized, axis=0, ignore_index=True)
# Now group by feature, only take features with count 3 and take the mean importance
overall_importance = overall_importance.groupby("feature").filter(lambda x: len(x) == 3)
overall_importance = overall_importance.groupby("feature").mean().reset_index()

# Get the final ranking
overall_importance = overall_importance.sort_values("importance", ascending=False)
overall_importance["rank"] = np.arange(1, len(overall_importance) + 1)

print(overall_importance)
print(overall_importance["feature"].values)
