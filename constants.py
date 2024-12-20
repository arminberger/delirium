FEATURE_GROUPS = {
    "setting_grp": [
        "(5:_setting)_Inpatient",
        "(5:_setting)_Outpatient",
        "(5:_setting)_Trauma",
    ],
    "surgical_specialty_grp": [
        "4:_surgical_speciality_General",
        "4:_surgical_speciality_Gyn/Obs",
        "4:_surgical_speciality_Neuro",
        "4:_surgical_speciality_Other",
        "4:_surgical_speciality_Plastic/breast",
        "4:_surgical_speciality_Trauma/Orth.",
        "4:_surgical_speciality_Urology",
        "4:_surgical_speciality_nan",
    ],
    "anesthesia_methods_grp": [
        "6:_anaesthesia_TCI (Target Controlled Infusion: drugs on level of brain receptors. Algorithm, so should be better than TIVA weight-based.)",
        "6:_anaesthesia_TIVA (Total intravenous anesthesia: based on only weight.)",
        #'6:_anaesthesia_SEVO (gas through the mask, inhalation - not intravenous like TCI/TIVA)',
        "6:_anaesthesia_General anaesthesia",
        "6:_anaesthesia_Sedation",
        "6:_anaesthesia_Spinal",
        #'6:_anaesthesia_Epidural',
        #'6:_anaesthesia_Upper extremity',
        #'6:_anaesthesia_Lower extremity'
    ],
    "patient_visual_aids_grp": [
        "19:_patient_uses_visual_aids_BI",
        "19a:_visual_aid_present_BI",
    ],
    "patient_hearing_aids_grp": [
        "20: patient_uses_hearing_aids_BI",
        "20a:_hearing_aid_present_BI",
    ],
    "patient_dentures_grp": [
        "21: patient_uses_dentures_BI",
        "21a: dentures_present_BI",
    ],
}
