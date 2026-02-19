# -*- coding: utf-8 -*-
"""Mavic 2 Zoom - Refactored for Deployment"""

import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Filter numerical columns
    df = df.select_dtypes(exclude=[object])

    # 2. Impute missing values
    df = df.ffill().bfill()
    df = df.apply(lambda col: col.fillna(col.mean()), axis=0)

    # 3. Select float/int only
    dc = df.select_dtypes(include=[float, int])

    # 4. EXACT features from 20Jan/mavic_2_zoom/preprocced_dataset.csv (48 cols)
    selected_features = [
        "IMU_ATTI(0):Longitude",
        "IMU_ATTI(0):Latitude",
        "IMU_ATTI(0):alti:D",
        "IMUCalcs(0):PosE:C",
        "IMUCalcs(0):PosD:C",
        "IMUCalcs(1):PosN:C",
        "IMUCalcs(1):PosE:C",
        "IMUCalcs(1):PosD:C",
        "IMUCalcs(0):Lat:C",
        "IMUCalcs(0):Long:C",
        "IMUCalcs(1):Lat:C",
        "IMUCalcs(1):Long:C",
        "IMU_ATTI(0):velN",
        "IMU_ATTI(0):velE",
        "IMU_ATTI(0):velD",
        "IMU_ATTI(0):accelX",
        "IMU_ATTI(0):accelY",
        "IMU_ATTI(0):accelZ",
        "IMU_ATTI(0):accelComposite:C",
        "IMU_ATTI(0):distanceTravelled:C",
        "IMU_ATTI(1):distanceTravelled:C",
        "IMU_ATTI(1):alti:D",
        "IMUCalcs(0):height:C",
        "IMUCalcs(1):height:C",
        "IMU_ATTI(0):roll:C",
        "IMU_ATTI(0):pitch:C",
        "IMU_ATTI(0):yaw:C",
        "IMU_ATTI(0):yaw360:C",
        "IMU_ATTI(0):magYaw:C",
        "IMU_ATTI(1):pitch:C",
        "IMUCalcs(0):totalGyroX:C",
        "IMUCalcs(0):totalGyroY:C",
        "IMUCalcs(0):totalGyroZ:C",
        "IMUCalcs(1):totalGyroX:C",
        "IMUCalcs(1):totalGyroY:C",
        "IMUCalcs(1):totalGyroZ:C",
        "ATTI_MINI0:roll:C",
        "ATTI_MINI0:pitch:C",
        "ATTI_MINI0:yaw:C",
        "IMU_ATTI(0):quatW:D",
        "IMU_ATTI(0):quatX:D",
        "IMU_ATTI(0):quatY:D",
        "IMU_ATTI(0):quatZ:D",
        "IMU_ATTI(1):quatY:D",
        "ATTI_MINI0:s_qx0",
        "ATTI_MINI0:s_qz0",
        "IMU_ATTI(0):directionOfTravel[true]:C",
        "Clock:Tick#",
    ]

    # 5. Add missing columns as 0
    for feat in selected_features:
        if feat not in dc.columns:
            dc[feat] = 0.0

    final_df = dc[selected_features].copy()
    final_df = final_df.fillna(0)
    return final_df
