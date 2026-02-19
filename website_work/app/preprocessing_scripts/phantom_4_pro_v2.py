# -*- coding: utf-8 -*-
"""Feature-Selection(vto_lab).ipynb - Refactored for Deployment"""

import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Filter numerical columns
    df = df.select_dtypes(exclude=[object])

    # 2. Drop specific columns (Stage 1)
    cols_to_drop_1 = [
        "IMU_ATTI(0):absoluteHeight:C",
        "IMU_ATTI(0):distanceHP:C",
        "RC_Info:appLost",
        "osd_data:lowVoltage",
        "ConvertDatV3",
        "4.3.0",
    ]
    df = df.drop(columns=cols_to_drop_1, errors="ignore")

    # 3. Drop specific columns (Stage 2)
    cols_to_drop_2 = [
        "IMU_ATTI(0):directionOfTravel[mag]:C",
        # 'IMU_ATTI(0):directionOfTravel[true]:C', # Dropped in 20Jan dataset
        "MVO:velZ",
        "MVO:posZ",
        "MVO:height",
        "usonic:usonic_h",
    ]
    df = df.drop(columns=cols_to_drop_2, errors="ignore")

    # 4. Impute missing values
    df = df.fillna(method="ffill").fillna(method="bfill")

    # 5. Fill remaining nulls with mean
    df = df.apply(lambda col: col.fillna(col.mean()), axis=0)

    # 6. Select float/int only
    dc = df.select_dtypes(include=[float, int])

    # 7. Define Selected Features (35 Features from 20Jan Dataset)
    selected_features = [
        "IMU_ATTI(0):Longitude",
        "IMU_ATTI(0):Latitude",
        "IMU_ATTI(0):alti:D",
        "IMUCalcs(0):PosN:C",
        "IMUCalcs(0):PosE:C",
        "IMUCalcs(0):PosD:C",
        "IMUCalcs(0):Lat:C",
        "IMUCalcs(0):Long:C",
        "IMU_ATTI(0):velN",
        "IMU_ATTI(0):velE",
        "IMU_ATTI(0):velD",
        "IMU_ATTI(0):accelX",
        "IMU_ATTI(0):accelY",
        "IMU_ATTI(0):accelZ",
        "IMU_ATTI(0):accelComposite:C",
        "IMU_ATTI(0):distanceTravelled:C",
        "IMUCalcs(0):height:C",
        "IMU_ATTI(0):roll:C",
        "IMU_ATTI(0):pitch:C",
        "IMU_ATTI(0):yaw:C",
        "IMU_ATTI(0):yaw360:C",
        "IMU_ATTI(0):magYaw:C",
        "IMUCalcs(0):totalGyroX:C",
        "IMUCalcs(0):totalGyroY:C",
        "IMUCalcs(0):totalGyroZ:C",
        "ATTI_MINI0:roll:C",
        "ATTI_MINI0:pitch:C",
        "ATTI_MINI0:yaw:C",
        "IMU_ATTI(0):quatW:D",
        "IMU_ATTI(0):quatX:D",
        "IMU_ATTI(0):quatY:D",
        "IMU_ATTI(0):quatZ:D",
        "ATTI_MINI0:s_qx0",
        "ATTI_MINI0:s_qz0",
        "Clock:Tick#",
    ]

    # 8. Filter to selected features
    # Ensure all exist (fill 0 if missing to prevent crash, though logic implies they exist)
    for feat in selected_features:
        if feat not in dc.columns:
            dc[feat] = 0.0

    new_df = dc[selected_features].copy()

    # 9. Final Nan Check
    new_df = new_df.fillna(0)

    return new_df
