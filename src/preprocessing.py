import pandas as pd

class DataTransformation:
    def __init__(self):
        pass

    def transform_data(self, df):
        try:
            # 1. Timestamp ని డేటైమ్ ఫార్మాట్‌లోకి మార్చడం
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values(by=['user_id', 'timestamp'])

            # 2. Rolling Window కోసం ఇండెక్స్‌ని సెట్ చేయడం (ముఖ్యమైన స్టెప్)
            df.set_index('timestamp', inplace=True)
            
            # 1 గంటలో జరిగిన బెట్స్ కౌంట్ (Velocity Check)
            # ఇక్కడ groupby యూజర్ మీద చేసి, టైమ్ విండోని లెక్కించాలి
            df['bet_count_1h'] = df.groupby('user_id')['bet_amount'].rolling('1h').count().values
            
            # ఇండెక్స్‌ని మళ్ళీ మామూలు స్థితికి తేవడం
            df.reset_index(inplace=True)

            # 3. High Stakes Indicator
            avg_bet = df.groupby('user_id')['bet_amount'].transform('mean')
            df['is_high_stake'] = (df['bet_amount'] > (3 * avg_bet)).astype(int)

            print("Feature Engineering completed successfully.")
            return df
        except Exception as e:
            # ఇక్కడ ఎర్రర్ వస్తే ఏం జరిగిందో క్లియర్ గా తెలుస్తుంది
            print(f"Error in Transformation: {e}")
            return None