import pandas as pd
import numpy as np
from scipy import stats
import argparse
import warnings
from collections import OrderedDict

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

COMMUNITY_AREA = 32  # Loop

# CTA Loop station coordinates
CTA_LOOP_STATIONS = {
    "Adams/Wabash":        (41.8797, -87.6260),
    "Clark/Lake":          (41.8858, -87.6309),
    "Harrison":            (41.8741, -87.6278),
    "Jackson/Dearborn":    (41.8783, -87.6319),
    "Jackson/State":       (41.8783, -87.6276),
    "LaSalle":             (41.8755, -87.6322),
    "LaSalle/Van Buren":   (41.8768, -87.6316),
    "Lake/State":          (41.8849, -87.6277),
    "Library":             (41.8766, -87.6281),
    "Madison/Wabash":      (41.8819, -87.6260),
    "Monroe/Dearborn":     (41.8808, -87.6292),
    "Monroe/State":        (41.8808, -87.6276),
    "Quincy/Wells":        (41.8788, -87.6337),
    "Randolph/Wabash":     (41.8847, -87.6260),
    "Roosevelt":           (41.8674, -87.6271),
    "State/Lake":          (41.8858, -87.6277),
    "Washington/Dearborn": (41.8833, -87.6292),
    "Washington/State":    (41.8833, -87.6276),
    "Washington/Wabash":   (41.8833, -87.6260),
    "Washington/Wells":    (41.8828, -87.6337),
}

# Groupings
VIOLENT_TYPES = ["BATTERY", "ASSAULT", "ROBBERY", "HOMICIDE", "CRIM SEXUAL ASSAULT"]
PROPERTY_TYPES = ["CRIMINAL DAMAGE", "BURGLARY", "MOTOR VEHICLE THEFT", "ARSON"]
STREET_LOCATIONS = ["STREET", "SIDEWALK", "ALLEY"]
RETAIL_LOCATIONS = ["SMALL RETAIL STORE", "DEPARTMENT STORE", "GROCERY FOOD STORE", "CONVENIENCE STORE", "RESTAURANT", "BAR OR TAVERN", "GAS STATION", "DRUG STORE", "COMMERCIAL / BUSINESS OFFICE"]
TRANSIT_KEYWORDS = ["CTA"]
PARKING_KEYWORDS = ["PARKING"]

SR_INFRASTRUCTURE = ["Street Light Out Complaint", "Street Light Pole Damage Complaint", "Traffic Signal Out Complaint", "Pothole in Street Complaint", "Sidewalk Inspection Request"]
SR_QUALITY_OF_LIFE = ["Graffiti Removal Request", "Rodent Baiting/Rat Complaint", "Sanitation Code Violation", "Street Cleaning Request", "Abandoned Vehicle Complaint"]
SR_BUSINESS_RELATED = ["Restaurant Complaint", "Business Complaints", "Building Violation"]

FOOD_KEYWORDS = ["food", "restaurant", "tavern", "consumption", "caterer", "mobile food"]
HIGH_VALUE_KEYWORDS = ["hotel", "bank", "securities", "insurance", "financial", "pawnbroker"]

# Target variable weights (BHS_WEIGHTS)
BHS_WEIGHTS = {
    "active_business_count": 0.25,
    "net_business_change": 0.20,
    "business_diversity_index": 0.15,
    "avg_license_age_months": 0.10,
    "high_value_biz_ratio": 0.05,
    "cta_monthly_ridership": 0.10,  
    "neg_infrastructure_issues": 0.10,  
    "neg_quality_of_life_issues": 0.05,  
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_block(block_str: str) -> str:
    if pd.isna(block_str): return "UNKNOWN"
    s = str(block_str).strip().upper()
    for sep in ["#", "SUITE", "STE", "UNIT", "FL ", "FLOOR"]:
        if sep in s: s = s[:s.index(sep)]
    s = " ".join(s.split())
    parts = s.split(" ", 1)
    if len(parts) >= 2 and parts[0].replace("X", "").isdigit():
        num = parts[0].replace("X", "0")
        try:
            s = f"{(int(num) // 100) * 100:04d}X {parts[1]}"
        except ValueError: pass
    return s.replace(" ", "_")

def haversine_meters(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def safe_normalize(series):
    mn, mx = series.min(), series.max()
    return pd.Series(0.5, index=series.index) if mx == mn else (series - mn) / (mx - mn)

def shannon_entropy(values):
    probs = values.value_counts(normalize=True)
    return -np.sum(probs * np.log(probs + 1e-10))

# ============================================================================
# LOADERS
# ============================================================================

def load_crime_data(path: str) -> pd.DataFrame:
    print(f"\n[1/7] Loading crime data: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    ca_col = [c for c in df.columns if "community" in c.lower() and "area" in c.lower()]
    if ca_col: df = df[pd.to_numeric(df[ca_col[0]], errors="coerce") == COMMUNITY_AREA]
    date_col = [c for c in df.columns if c.lower() == "date"][0]
    df["datetime"] = pd.to_datetime(df[date_col], format="mixed", errors="coerce")
    df = df.dropna(subset=["datetime"])
    df["year_month"], df["hour"], df["day_of_week"], df["day"] = df["datetime"].dt.to_period("M"), df["datetime"].dt.hour, df["datetime"].dt.dayofweek, df["datetime"].dt.date
    df["block_id"] = df[[c for c in df.columns if c.lower() == "block"][0]].apply(normalize_block)
    df["primary_type"] = df.get("Primary Type", pd.Series("UNKNOWN")).str.strip().str.upper()
    df["location_description"] = df.get("Location Description", pd.Series("UNKNOWN")).str.strip().str.upper()
    df["arrest"] = df.get("Arrest", pd.Series(False)).astype(str).str.upper().isin(["TRUE", "Y", "1"])
    df["domestic"] = df.get("Domestic", pd.Series(False)).astype(str).str.upper().isin(["TRUE", "Y", "1"])
    df["latitude"], df["longitude"] = pd.to_numeric(df.get("Latitude"), errors="coerce"), pd.to_numeric(df.get("Longitude"), errors="coerce")
    return df

def load_business_data(path: str) -> pd.DataFrame:
    print(f"\n[2/7] Loading business licenses: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    ca_col = [c for c in df.columns if "community" in c.lower() and "area" in c.lower() and "name" not in c.lower()]
    if ca_col: df = df[pd.to_numeric(df[ca_col[0]], errors="coerce") == COMMUNITY_AREA]
    for col, target in [("LICENSE TERM START DATE", "start_date"), ("LICENSE TERM EXPIRATION DATE", "end_date"), ("DATE ISSUED", "date_issued")]:
        if col in df.columns: df[target] = pd.to_datetime(df[col], format="mixed", errors="coerce")
    df["license_status"] = df.get("LICENSE STATUS", pd.Series("AAI")).fillna("AAI").str.strip().str.upper()
    df["biz_type"] = df.get("LICENSE DESCRIPTION", pd.Series("UNKNOWN")).fillna("UNKNOWN").str.strip()
    df["block_id"] = df.get("ADDRESS", pd.Series("UNKNOWN")).apply(normalize_block)
    df["latitude"], df["longitude"] = pd.to_numeric(df.get("LATITUDE"), errors="coerce"), pd.to_numeric(df.get("LONGITUDE"), errors="coerce")
    return df

def load_311_data(path: str) -> pd.DataFrame:
    print(f"\n[3/7] Loading 311 service requests: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    if "COMMUNITY_AREA" in df.columns: df = df[pd.to_numeric(df["COMMUNITY_AREA"], errors="coerce") == COMMUNITY_AREA]
    df["datetime"] = pd.to_datetime(df["CREATED_DATE"], format="mixed", errors="coerce")
    df = df.dropna(subset=["datetime"])
    df["year_month"] = df["datetime"].dt.to_period("M")
    df["full_address"] = df.get("STREET_NUMBER", "").astype(str) + " " + df.get("STREET_DIRECTION", "").fillna("").astype(str) + " " + df.get("STREET_NAME", "").fillna("").astype(str) + " " + df.get("STREET_TYPE", "").fillna("").astype(str)
    df["block_id"] = df["full_address"].apply(normalize_block)
    df["sr_type"] = df.get("SR_TYPE", pd.Series("UNKNOWN")).fillna("UNKNOWN").str.strip()
    return df

def load_cta_data(path: str) -> pd.DataFrame:
    print(f"\n[4/7] Loading CTA ridership: {path}")
    df = pd.read_csv(path, low_memory=False)
    df = df[df["stationame"].isin(CTA_LOOP_STATIONS.keys())]
    df["year_month"] = pd.to_datetime(df["month_beginning"], format="mixed", errors="coerce").dt.to_period("M")
    df = df.dropna(subset=["year_month"])
    for col in ["avg_weekday_rides", "avg_saturday_rides", "avg_sunday-holiday_rides", "monthtotal"]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce").fillna(0)
    return df

# ============================================================================
# FEATURE COMPUTERS
# ============================================================================

def compute_crime_features(crime_df: pd.DataFrame) -> pd.DataFrame:
    print("\n[5/7] Computing crime features...")
    g = crime_df.groupby(["block_id", "year_month"])
    vol = g.agg(
        total_crimes=("primary_type", "size"),
        theft_count=("primary_type", lambda x: (x == "THEFT").sum()),
        violent_crime_count=("primary_type", lambda x: x.isin(VIOLENT_TYPES).sum()),
        property_crime_count=("primary_type", lambda x: x.isin(PROPERTY_TYPES).sum()),
        fraud_count=("primary_type", lambda x: (x == "DECEPTIVE PRACTICE").sum()),
        narcotics_count=("primary_type", lambda x: (x == "NARCOTICS").sum()),
        weapons_count=("primary_type", lambda x: (x == "WEAPONS VIOLATION").sum()),
        arrest_count=("arrest", "sum"),
        domestic_count=("domestic", "sum"),
    ).reset_index()
    vol["arrest_rate"] = (vol["arrest_count"] / vol["total_crimes"].replace(0, np.nan)).fillna(0)
    vol["violent_to_total_ratio"] = (vol["violent_crime_count"] / vol["total_crimes"].replace(0, np.nan)).fillna(0)
    vol = vol.merge(g["primary_type"].apply(shannon_entropy).reset_index(name="crime_diversity_index"), on=["block_id", "year_month"], how="left")
    
    temp = g.agg(
        night_crime_count=("hour", lambda x: ((x >= 20) | (x < 6)).sum()),
        weekend_crime_count=("day_of_week", lambda x: x.isin([5, 6]).sum()),
        peak_crime_hour=("hour", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 12),
        crime_hour_spread=("hour", lambda x: x.std() if len(x) > 1 else 0),
    ).reset_index()
    
    bh = crime_df[(crime_df["hour"] >= 9) & (crime_df["hour"] < 17) & (crime_df["day_of_week"] < 5)]
    temp = temp.merge(bh.groupby(["block_id", "year_month"]).size().reset_index(name="business_hours_crime"), on=["block_id", "year_month"], how="left")
    temp["business_hours_crime"] = temp["business_hours_crime"].fillna(0).astype(int)
    temp = temp.merge(vol[["block_id", "year_month", "total_crimes"]], on=["block_id", "year_month"])
    temp["night_crime_ratio"] = (temp["night_crime_count"] / temp["total_crimes"].replace(0, np.nan)).fillna(0)
    temp["weekend_crime_ratio"] = (temp["weekend_crime_count"] / temp["total_crimes"].replace(0, np.nan)).fillna(0)
    temp = temp.drop(columns=["total_crimes"])
    
    daily = crime_df.groupby(["block_id", "year_month", "day"]).size().reset_index(name="dc")
    temp = temp.merge(daily.groupby(["block_id", "year_month"])["dc"].max().reset_index(name="max_daily_crimes"), on=["block_id", "year_month"], how="left")
    temp["max_daily_crimes"], temp["crime_hour_spread"] = temp["max_daily_crimes"].fillna(0).astype(int), temp["crime_hour_spread"].fillna(0)
    
    loc = g.agg(
        street_crime_count=("location_description", lambda x: x.isin(STREET_LOCATIONS).sum()),
        retail_crime_count=("location_description", lambda x: x.isin(RETAIL_LOCATIONS).sum()),
        transit_crime_count=("location_description", lambda x: x.str.contains("|".join(TRANSIT_KEYWORDS), na=False).sum()),
        parking_crime_count=("location_description", lambda x: x.str.contains("|".join(PARKING_KEYWORDS), na=False).sum()),
        indoor_crime_count=("location_description", lambda x: x.isin(["APARTMENT", "RESIDENCE", "OFFICE", "HOTEL/MOTEL", "BANK"]).sum()),
        location_diversity=("location_description", "nunique"),
    ).reset_index()
    loc = loc.merge(vol[["block_id", "year_month", "total_crimes"]], on=["block_id", "year_month"])
    loc["street_crime_ratio"] = (loc["street_crime_count"] / loc["total_crimes"].replace(0, np.nan)).fillna(0)
    loc["indoor_crime_ratio"] = (loc["indoor_crime_count"] / loc["total_crimes"].replace(0, np.nan)).fillna(0)
    loc = loc.drop(columns=["total_crimes", "street_crime_count", "indoor_crime_count"])
    
    return vol.merge(temp, on=["block_id", "year_month"], how="outer").merge(loc, on=["block_id", "year_month"], how="outer")

def compute_business_features(biz_df: pd.DataFrame, date_range: pd.PeriodIndex) -> pd.DataFrame:
    print("\n[6/7] Computing business features...")
    biz_df = biz_df[biz_df["license_status"].isin(["AAI", "AAC"])].copy()
    biz_df["is_food"] = biz_df["biz_type"].str.lower().str.contains("|".join(FOOD_KEYWORDS), na=False)
    biz_df["is_high_value"] = biz_df["biz_type"].str.lower().str.contains("|".join(HIGH_VALUE_KEYWORDS), na=False)
    
    records = []
    for i, period in enumerate(date_range):
        pstart, pend = period.start_time, period.end_time
        active_mask = pd.Series(True, index=biz_df.index)
        if "start_date" in biz_df.columns: active_mask &= (biz_df["start_date"] <= pend) | biz_df["start_date"].isna()
        if "end_date" in biz_df.columns: active_mask &= (biz_df["end_date"] >= pstart) | biz_df["end_date"].isna()
        
        active = biz_df[active_mask].copy()
        
        new_mask = (biz_df["date_issued"] >= pstart) & (biz_df["date_issued"] <= pend) if "date_issued" in biz_df.columns else ((biz_df["start_date"] >= pstart) & (biz_df["start_date"] <= pend) if "start_date" in biz_df.columns else pd.Series(False, index=biz_df.index))
        new_counts = biz_df[new_mask].groupby("block_id").size()
        
        exp_counts = biz_df[(biz_df["end_date"] >= pstart) & (biz_df["end_date"] <= pend)].groupby("block_id").size() if "end_date" in biz_df.columns else pd.Series(dtype=int)
        
        grouped = active.groupby("block_id")
        monthly_stats = grouped.agg(active_business_count=("biz_type", "size"), food_business_ratio=("is_food", "mean"), high_value_biz_ratio=("is_high_value", "mean")).reset_index()
        monthly_stats["business_diversity_index"] = monthly_stats["block_id"].map(grouped["biz_type"].apply(lambda x: shannon_entropy(x) if len(x) > 1 else 0))
        
        if "start_date" in active.columns:
            active["age_months"] = (pend - active["start_date"]).dt.days / 30.44
            monthly_stats["avg_license_age_months"] = monthly_stats["block_id"].map(grouped["age_months"].mean()).fillna(0)
        else:
            monthly_stats["avg_license_age_months"] = 0
            
        monthly_stats["new_licenses"] = monthly_stats["block_id"].map(new_counts).fillna(0).astype(int)
        monthly_stats["expired_licenses"] = monthly_stats["block_id"].map(exp_counts).fillna(0).astype(int)
        monthly_stats["net_business_change"] = monthly_stats["new_licenses"] - monthly_stats["expired_licenses"]
        monthly_stats["year_month"] = period
        records.append(monthly_stats)
        
    result = pd.concat(records, ignore_index=True)
    for col, dec in [("business_diversity_index", 4), ("food_business_ratio", 4), ("avg_license_age_months", 2), ("high_value_biz_ratio", 4)]: result[col] = result[col].round(dec)
    return result

def compute_311_features(sr_df: pd.DataFrame) -> pd.DataFrame:
    print("\n  Computing 311 features...")
    g = sr_df.groupby(["block_id", "year_month"])
    result = g.agg(total_311_requests=("sr_type", "size")).reset_index()
    
    for label, mask_list in [("infrastructure_issues", SR_INFRASTRUCTURE), ("quality_of_life_issues", SR_QUALITY_OF_LIFE), ("business_complaints", SR_BUSINESS_RELATED)]:
        result = result.merge(sr_df[sr_df["sr_type"].isin(mask_list)].groupby(["block_id", "year_month"]).size().reset_index(name=label), on=["block_id", "year_month"], how="left")
        
    for label, keyword in [("streetlight_issues", "Street Light"), ("graffiti_reports", "Graffiti")]:
        result = result.merge(sr_df[sr_df["sr_type"].str.contains(keyword, case=False, na=False)].groupby(["block_id", "year_month"]).size().reset_index(name=label), on=["block_id", "year_month"], how="left")
        
    if "CLOSED_DATE" in sr_df.columns:
        sr_df["closed_dt"] = pd.to_datetime(sr_df["CLOSED_DATE"], format="mixed", errors="coerce")
        sr_df["resolution_days"] = ((sr_df["closed_dt"] - sr_df["datetime"]).dt.total_seconds() / 86400).clip(0, 365)
        result = result.merge(sr_df.groupby(["block_id", "year_month"])["resolution_days"].mean().reset_index(name="avg_resolution_days"), on=["block_id", "year_month"], how="left")
    else:
        result["avg_resolution_days"] = np.nan
        
    result.fillna({col: 0 for col in ["infrastructure_issues", "quality_of_life_issues", "business_complaints", "streetlight_issues", "graffiti_reports", "avg_resolution_days"]}, inplace=True)
    return result

def compute_cta_features(cta_df: pd.DataFrame, block_centroids: pd.DataFrame) -> pd.DataFrame:
    print("\n  Computing CTA ridership features...")
    area_ridership = cta_df.groupby("year_month").agg(cta_total_loop_ridership=("monthtotal", "sum"), cta_avg_weekday_rides=("avg_weekday_rides", "mean"), cta_avg_weekend_rides=("avg_saturday_rides", "mean")).reset_index()
    
    nearest_station = {}
    for _, row in block_centroids.iterrows():
        blk, blat, blon = row["block_id"], row["lat_centroid"], row["lon_centroid"]
        if pd.isna(blat) or pd.isna(blon):
            nearest_station[blk] = None
            continue
        nearest_station[blk] = min(CTA_LOOP_STATIONS.items(), key=lambda x: haversine_meters(blat, blon, x[1][0], x[1][1]))[0]
        
    block_station_map = pd.DataFrame([{"block_id": k, "nearest_cta_station": v} for k, v in nearest_station.items()])
    block_station_map["dist_to_nearest_cta"] = block_station_map.apply(lambda r: haversine_meters(block_centroids.set_index("block_id").loc[r["block_id"], "lat_centroid"], block_centroids.set_index("block_id").loc[r["block_id"], "lon_centroid"], *CTA_LOOP_STATIONS[r["nearest_cta_station"]]) if r["nearest_cta_station"] else np.nan, axis=1)
    
    return area_ridership, cta_df.groupby(["stationame", "year_month"]).agg(nearest_station_ridership=("monthtotal", "sum")).reset_index(), block_station_map

def compute_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n  Computing lag features...")
    df = df.sort_values(["block_id", "year_month"])
    for col, lags in [("total_crimes", [1, 3]), ("active_business_count", [1])]:
        if col in df.columns:
            for lag in lags: df[f"{'crime' if 'crime' in col else 'biz'}_count_lag{lag}"] = df.groupby("block_id")[col].shift(lag)
            
    if "total_crimes" in df.columns:
        df["crime_rolling_3m_avg"] = df.groupby("block_id")["total_crimes"].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df["crime_rolling_3m_std"] = df.groupby("block_id")["total_crimes"].transform(lambda x: x.rolling(3, min_periods=1).std()).fillna(0)
    if "crime_count_lag1" in df.columns:
        df["crime_mom_change"] = np.where(df["crime_count_lag1"] > 0, (df["total_crimes"] - df["crime_count_lag1"]) / df["crime_count_lag1"], 0)
        
    def trend_slope(series): return pd.Series([np.nan if i < 2 or np.any(np.isnan(series.values[i-2:i+1])) else stats.linregress(range(3), series.values[i-2:i+1]).slope for i in range(len(series))], index=series.index)
    for col, name in [("total_crimes", "crime_trend_3m"), ("violent_crime_count", "violent_trend_3m")]:
        if col in df.columns: df[name] = df.groupby("block_id")[col].transform(trend_slope)
    if "total_311_requests" in df.columns: df["sr_count_lag1"] = df.groupby("block_id")["total_311_requests"].shift(1)
    return df

def compute_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n  Computing spatial features...")
    if "lat_centroid" in df.columns and df["lat_centroid"].notna().any():
        RADIUS = 0.002
        neighbor_avgs = []
        for ym in df["year_month"].unique():
            md = df[df["year_month"] == ym].copy()
            valid_mask = md["lat_centroid"].notna() & md["lon_centroid"].notna()
            if valid_mask.sum() == 0:
                neighbor_avgs.extend([0] * len(md))
                continue
            lats, lons, crimes = md["lat_centroid"].values, md["lon_centroid"].values, md["total_crimes"].values
            lat_diff = np.abs(lats[:, np.newaxis] - lats[np.newaxis, :])
            lon_diff = np.abs(lons[:, np.newaxis] - lons[np.newaxis, :])
            within_radius = (lat_diff <= RADIUS) & (lon_diff <= RADIUS)
            np.fill_diagonal(within_radius, False)
            neighbor_avgs.extend([crimes[within_radius[i]].mean() if valid_mask.iloc[i] and within_radius[i].any() else 0 for i in range(len(md))])
        df["neighbor_avg_crime"] = neighbor_avgs
    else:
        df["neighbor_avg_crime"] = 0

    df["crime_relative_to_neighbors"] = np.where(df["neighbor_avg_crime"] > 0, df["total_crimes"] / df["neighbor_avg_crime"], 1)
    month_num = pd.to_datetime(df["year_month"].astype(str)).dt.month
    df["month_sin"], df["month_cos"] = np.sin(2 * np.pi * month_num / 12), np.cos(2 * np.pi * month_num / 12)
    df["crimes_per_business"] = np.where(df["active_business_count"] > 0, df["total_crimes"] / df["active_business_count"], df["total_crimes"])
    return df

def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    print("\n  Computing target: Business Health Score (BHS)...")
    components = {}
    for col, weight in BHS_WEIGHTS.items():
        if col.startswith("neg_"):
            actual_col = col.replace("neg_", "")
            if actual_col in df.columns: components[col] = (1 - safe_normalize(df[actual_col].fillna(0))) * weight
        elif col == "cta_monthly_ridership" and "cta_total_loop_ridership" in df.columns:
            components[col] = safe_normalize(df["cta_total_loop_ridership"].fillna(0)) * weight
        elif col in df.columns:
            components[col] = safe_normalize(df[col].fillna(0)) * weight

    df["business_health_score"] = (sum(components.values()) * 100).clip(0, 100).round(2) if components else 50.0
    return df

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(crime_path, business_path, sr_path, cta_path, output_path):
    print("=" * 70 + "\n  LOOP PULSE v2 — Feature Engineering Pipeline\n" + "=" * 70)
    
    crime_df = load_crime_data(crime_path)
    biz_df = load_business_data(business_path)
    sr_df = load_311_data(sr_path)
    cta_df = load_cta_data(cta_path)

    min_period, max_period = crime_df["year_month"].min(), crime_df["year_month"].max()
    date_range = pd.period_range(start=min_period, end=max_period, freq="M")
    
    crime_df = crime_df[(crime_df["year_month"] >= min_period) & (crime_df["year_month"] <= max_period)]
    sr_df = sr_df[(sr_df["year_month"] >= min_period) & (sr_df["year_month"] <= max_period)]

    centroids = crime_df.dropna(subset=["latitude", "longitude"]).groupby("block_id").agg(lat_centroid=("latitude", "mean"), lon_centroid=("longitude", "mean")).reset_index()

    crime_features = compute_crime_features(crime_df)
    biz_features = compute_business_features(biz_df, date_range)
    sr_features = compute_311_features(sr_df)
    area_ridership, station_monthly, block_station_map = compute_cta_features(cta_df, centroids)

    print("\n[7/7] Merging all features...")
    merged = crime_features.copy()
    merged = merged.merge(biz_features, on=["block_id", "year_month"], how="outer")
    merged = merged.merge(sr_features, on=["block_id", "year_month"], how="outer")
    merged = merged.merge(area_ridership, on=["year_month"], how="left")
    merged = merged.merge(block_station_map[["block_id", "nearest_cta_station", "dist_to_nearest_cta"]], on="block_id", how="left")
    merged = merged.merge(station_monthly.rename(columns={"stationame": "nearest_cta_station"}), on=["nearest_cta_station", "year_month"], how="left")
    merged = merged.merge(centroids, on="block_id", how="left")

    num_cols = merged.select_dtypes(include=[np.number]).columns
    merged[num_cols] = merged[num_cols].fillna(0)

    merged = compute_lag_features(merged)
    merged = compute_spatial_features(merged)
    merged = compute_target(merged)

    id_cols = ["block_id", "year_month", "lat_centroid", "lon_centroid", "nearest_cta_station"]
    target_cols = ["business_health_score"]
    feature_cols = [c for c in merged.columns if c not in id_cols + target_cols]
    
    merged = merged[id_cols + sorted(feature_cols) + target_cols]
    merged["year_month"] = merged["year_month"].astype(str)
    merged = merged[merged["total_crimes"] > 0]
    
    merged.to_csv(output_path, index=False)
    print(f"\nSaved {merged.shape[0]} rows to {output_path}")

if __name__ == "__main__":
    run_pipeline(
        crime_path="Crimes_-_2001_to_Present_20260228.csv",
        business_path="Business_Licenses.csv",
        sr_path="311_Service_Requests_20260228.csv",
        cta_path="CTA_Loop_Stations.csv", # Ensure this matches your local CSV name
        output_path="loop_pulse_features2.csv"
    )