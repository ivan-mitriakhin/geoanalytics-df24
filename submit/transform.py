import pandas as pd
import numpy as np
import h3

class BaseTransform:
    def __init__(self, filepath=None):
        if filepath:
            self.load(filepath)
        pass

    def fit(self, transactions):
        pass

    def transform(self, transactions):
        return transactions

    def save(self, filepath):
        pass

    def load(self, filepath):
        pass

class SimpleFeaturesTransform(BaseTransform):
    def __init__(self, filepath=None):
        if filepath:
            self.load(filepath)
        pass

    def fit(self, transactions, hexses_data):
        # self.mcc_codes = set(transactions.mcc_code.unique())
        # self.datetime_ids = set(transactions.datetime_id.unique())
        self.hexses_data = set(hexses_data)
        self.suburbs = set(transactions.suburb.unique())
        chunk = dict()
        # for mcc_code in self.mcc_codes:
        #     chunk[f"mcc_{ mcc_code }_count"] = 0
        #     chunk[f"mcc_{ mcc_code }_sum"] = 0
        # for datetime_id in self.datetime_ids:
        #     chunk[f"dt_{ datetime_id }_count"] = 0
        #     chunk[f"dt_{ datetime_id }_sum"] = 0
        for h3_09 in self.hexses_data:
            chunk[f"hex_{ h3_09 }_count"] = 0
        for suburb in self.suburbs:
            chunk[f"{ suburb }_count"] = 0
        self.template = chunk

    def transform(self, transactions):
        features = []
        row_labels = []
        gb = transactions.groupby(by="customer_id")
        for customer_id, group in gb:
            row_labels.append(customer_id)
            chunk = self.template.copy()

            # chunk['lat_median'] = group['lat'].median()
            # chunk['lng_median'] = group['lng'].median()

            # for mcc_code, subgroup in group.groupby(by='mcc_code'):
            #     if mcc_code in self.mcc_codes:
            #         chunk[f"mcc_{ mcc_code }_count"] = subgroup["count"].sum()
            #         chunk[f"mcc_{ mcc_code }_sum"] = subgroup["sum"].sum()

            # for datetime_id, subgroup in group.groupby(by='datetime_id'):
            #     if datetime_id in self.datetime_ids:
            #         chunk[f"dt_{ datetime_id }_count"] = subgroup["count"].sum()
            #         chunk[f"dt_{ datetime_id }_count"] = 0

            for h3_09, subgroup in group.groupby(by='h3_09'):
                if h3_09 in self.hexses_data:
                    chunk[f"hex_{ h3_09 }_count"] = subgroup['count'].sum()
            
            for suburb, subgroup in group.groupby(by='suburb'):
                if suburb in self.suburbs:
                    chunk[f"{ suburb }_count"] = subgroup['count'].sum()


            features.append(chunk)
        return pd.DataFrame(features, index=row_labels)

    def save(self, filepath):
        pass

    def load(self, filepath):
        pass

def extract_hexses_latlng(hexses_data, hexses_target):
    """
    Creates a dictionary of a form h3_index:(lat, lng).
    """
    hexses_latlng = dict()

    for hex in hexses_data:
        latlng = h3.h3_to_geo(hex)
        hexses_latlng[hex] = latlng

    for hex in hexses_target:
        latlng = h3.h3_to_geo(hex)
        hexses_latlng[hex] = latlng

    return hexses_latlng


def generate_home_features(transactions, hexses_data, hexses_suburb, hexses_latlng):
    city_center = '8911aa7abcbffff'

    # Create estimated customer homes dataframe for each customer.
    customers_homes = (transactions[transactions['mcc_code'] == 13]
    .groupby(['customer_id', 'h3_09', 'lat', 'lng'], as_index=False)['count'].sum()
    .sort_values(by=['customer_id', 'count'], ascending=False)
    .groupby('customer_id')
    .first()
    .reset_index())
    customers_no_homes = set(transactions.customer_id.unique()) - set(customers_homes.customer_id)
    estimated_customers_homes = (transactions[transactions['customer_id'].isin(customers_no_homes)]
                            .groupby(['customer_id', 'h3_09', 'lat', 'lng'], as_index=False)['count'].sum()
                            .sort_values(by=['customer_id', 'count'], ascending=False)
                            .groupby('customer_id', as_index=False)
                            .first())
    customers_homes = pd.concat([customers_homes, estimated_customers_homes])

    # Home to center/median_latlng distances for each customer.
    customers_homes['center_lat'] = hexses_latlng[city_center][0]
    customers_homes['center_lng'] = hexses_latlng[city_center][1]
    customers_homes['home_center_dist'] = customers_homes.apply(lambda x: h3.point_dist((x['lat'], x['lng']),
                                                                                        (x['center_lat'], x['center_lng']),
                                                                                        unit='km'), axis=1)
    lats = []
    lngs = []
    for h3_09 in hexses_data:
        lats.append(hexses_latlng[h3_09][0])
        lngs.append(hexses_latlng[h3_09][1])

    customers_homes['median_lat'] = np.median(lats)
    customers_homes['median_lng'] = np.median(lngs)

    customers_homes['home_median_latlng_dist'] = customers_homes.apply(lambda x: h3.point_dist((x['lat'], x['lng']),
                                                                                        (x['median_lat'], x['median_lng']),
                                                                                        unit='km'), axis=1)

    homes_distances = customers_homes[['customer_id', 'home_center_dist', 'home_median_latlng_dist']]

    # Different statistics for transactions done at home for each customer.
    homes_transactions = transactions.merge(customers_homes[['customer_id', 'h3_09']], on=['customer_id', 'h3_09'], how='right')
    homes_transactions = (homes_transactions
                        .groupby('customer_id')
                        .agg({'sum': 'sum', 'avg':'median', 'max': 'max', 'min': 'min', 'count': 'sum'})
                        .add_prefix('home_')
                        .reset_index())
    
    # Defines which suburb of the city each customer's home in, then one hot encodes it.
    homes_suburbs = customers_homes[['customer_id', 'h3_09']]
    homes_suburbs['home_suburb'] = homes_suburbs.apply(lambda x: hexses_suburb[x['h3_09']], axis=1)
    homes_suburbs = pd.get_dummies(homes_suburbs, columns=['home_suburb']).drop('h3_09', axis=1)

    # Gather all the home-related features.
    home_features = homes_suburbs.merge(homes_distances, on='customer_id').merge(homes_transactions, on='customer_id')

    return home_features



def generate_features(transactions, hexses_data, hexses_target, hexses_suburb):
    hexses_latlng = extract_hexses_latlng(hexses_data, hexses_target)

    all_hexses = list(set(hexses_data) | set(hexses_target))
    all_hexses = pd.DataFrame({"h3_09" : all_hexses})
    all_hexses[['lat', 'lng']] = all_hexses['h3_09'].apply(lambda x: pd.Series(hexses_latlng[x]))
    all_hexses['suburb'] = all_hexses['h3_09'].apply(lambda x: hexses_suburb[x])

    transactions = pd.merge(transactions, all_hexses, on="h3_09")

    transformer = SimpleFeaturesTransform()
    transformer.fit(transactions, hexses_data)
    features = transformer.transform(transactions)
    features = features.reset_index(names='customer_id')
    
    home_features = generate_home_features(transactions, hexses_data, hexses_suburb, hexses_latlng)

    features = features.merge(home_features, on='customer_id')

    return features
    