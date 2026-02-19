# -*- coding: utf-8 -*-
"""Matrice 210 - Refactored for Deployment"""

import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Filter numerical columns
    df = df.select_dtypes(exclude=[object])

    # 2. Impute missing values
    df = df.ffill().bfill()
    df = df.apply(lambda col: col.fillna(col.mean()), axis=0)

    # 3. Select float/int only
    dc = df.select_dtypes(include=[float, int])

    # 4. EXACT features from 20Jan/matrice_210/preprocced_dataset.csv (32 cols)
    # NOTE: Matrice 210 targets are GPS:Long, GPS:Lat, GPS:heightMSL
    selected_features = [
        "Clock:Tick#",
        "GPS:Long",
        "GPS:Lat",
        "GPS:heightMSL",
        "IMUCalcs(0):PosN:C",
        "IMUCalcs(0):PosE:C",
        "IMUCalcs(0):PosD:C",
        "IMU_ATTI(0):relativeHeight:C",
        "IMUCalcs(0):velN:C",
        "IMUCalcs(0):velE:C",
        "IMUCalcs(0):velD:C",
        "IMU_ATTI(0):velN",
        "IMU_ATTI(0):velE",
        "IMU_ATTI(0):velD",
        "IMU_ATTI(0):velComposite:C",
        "IMU_ATTI(0):accelX",
        "IMU_ATTI(0):accelY",
        "IMU_ATTI(0):accelZ",
        "IMU_ATTI(0):accelComposite:C",
        "IMU_ATTI(0):roll:C",
        "IMU_ATTI(0):pitch:C",
        "IMU_ATTI(0):yaw:C",
        "IMU_ATTI(0):yaw360:C",
        "IMU_ATTI(0):magYaw:C",
        "IMU_ATTI(0):gyroX",
        "IMU_ATTI(0):gyroY",
        "IMU_ATTI(0):gyroZ",
        "IMU_ATTI(0):gyroComposite:C",
        "AirSpeed:windSpeed",
        "AirSpeed:windDirection",
        "AirSpeed:windN",
        "AirSpeed:windE",
    ]

    # 5. Add missing columns as 0
    for feat in selected_features:
        if feat not in dc.columns:
            dc[feat] = 0.0

    final_df = dc[selected_features].copy()
    final_df = final_df.fillna(0)
    return final_df
