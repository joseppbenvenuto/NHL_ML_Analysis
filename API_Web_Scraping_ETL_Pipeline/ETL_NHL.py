# Import libraries
import psycopg2 as ps
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings('ignore')

import os
import re
import time
import glob
from datetime import date

from SQL_Queries_NHL import *


###########################################################################################################################################
# NEW CODE BLOCK - Process season and playoff data
###########################################################################################################################################

def process_data(start_date, end_date):
    '''
    - Get all season and playoff data from data directory where API and web scrape data is stored
    - Change columns names to match nhldb schema
    - Create stats columns
    - Change data types
    - Create unique ids for teams, time, and season stats
    - Left join season and playoff data
    '''
    # Import NHL team season stats data frame
    # Get file dynamically on any windows machine
    path = r'C:\**\API_Web_Scraping_ETL_Pipeline\Data'
    path = glob.glob(path, recursive = True)
    path = path[0]

    nhl_csv = []
    for i in os.listdir(path):
        # Select file with current year minus 1 and seasons in the file name
        if os.path.isfile(os.path.join(path,i)) and start_date and end_date and 'Season' in i:
            nhl_csv.append(i)

    # Convert csv file to data frame
    nhl_df = pd.read_csv(path + '\\' + str(nhl_csv[0]))

    # Failed shots per game feature
    nhl_df['failedShotsPerGame'] = nhl_df['shotsPerGame'] * (1 - (nhl_df['shootingPctg'] / 100))

    # Saves per game feature
    nhl_df['savesPerGame'] = nhl_df['shotsAllowed'] * nhl_df['savePctg']

    # Tie games feature
    nhl_df['ties'] = nhl_df['gamesPlayed'] - nhl_df['wins'] - nhl_df['losses']

    # Powerplay Efficiency feature
    nhl_df['powerPlayEfficiency'] = ((nhl_df['powerPlayGoals'] - nhl_df['powerPlayGoalsAgainst']) /
                                      nhl_df['powerPlayOpportunities'])

    # Adjusted wins calculated as total wins plus ties where two ties equate to one game won
    nhl_df['adjWins'] = nhl_df['wins'] + (nhl_df['ties'] / 2)

    # Move ajusted wins feature to the end of the data frame
    nhl_df = nhl_df[[c for c in nhl_df if c not in ['adjWins']] + ['adjWins']]

    # rename columns to match sql table schemas
    nhl_df.columns = [
        'games_played', 
        'wins', 
        'losses', 
        'ot', 
        'pts', 
        'pt_pctg', 
        'goals_per_game',
        'goals_against_per_game', 
        'evgga_ratio', 
        'power_play_percentage',
        'power_play_goals',
        'power_play_goals_against', 
        'power_play_opportunities',
        'penalty_kill_percentage',
        'shots_per_game', 
        'shots_allowed',
        'win_score_first',
        'win_opp_score_first', 
        'win_lead_first_per',
        'win_lead_second_per', 
        'win_outshoot_opp', 
        'win_outshot_by_opp',
        'face_offs_taken',
        'face_offs_won', 
        'face_offs_lost', 
        'face_off_win_percentage',
        'shooting_pctg', 
        'save_pctg', 
        'id', 
        'team', 
        'season_year_range',
        'failed_shots_per_game', 
        'saves_per_game', 
        'ties', 
        'power_play_efficiency',
        'adjWins'
    ]


    # Import NHL team playoff stats data frame
    nhl_csv = []
    for i in os.listdir(path):
        # Select file with current year minus 1 and playoff in the file name
        if os.path.isfile(os.path.join(path,i)) and start_date and end_date and 'Playoff' in i:
            nhl_csv.append(i)

    nhl_playoff_df = pd.read_csv(path + '\\' + str(nhl_csv[0]))
    nhl_playoff_df['made_playoffs'] = 1


    # Full data set merge
    nhl_full_df = nhl_df.merge(
        nhl_playoff_df,
        how = 'left',
        left_on = ['id','season_year_range'],
        right_on = ['id','playoff_year_range']
    )

    # Fill null values with 0 and change data types from float to int
    nhl_full_df = nhl_full_df.fillna(0)
    nhl_full_df['playoff_year_range'] = nhl_full_df['playoff_year_range'].astype(int)
    nhl_full_df['made_playoffs'] = nhl_full_df['made_playoffs'].astype(int)
    nhl_full_df['power_play_goals'] = nhl_full_df['power_play_goals'].astype(int)
    nhl_full_df['power_play_goals_against'] = nhl_full_df['power_play_goals_against'].astype(int)
    nhl_full_df['power_play_opportunities'] = nhl_full_df['power_play_opportunities'].astype(int)
    nhl_full_df['face_offs_taken'] = nhl_full_df['face_offs_taken'].astype(int)
    nhl_full_df['face_offs_won'] = nhl_full_df['face_offs_won'].astype(int)
    nhl_full_df['face_offs_lost'] = nhl_full_df['face_offs_lost'].astype(int)

    # Create unique id for season stats
    nhl_full_df = nhl_full_df.reset_index()
    nhl_full_df['season_stats_id'] = nhl_full_df['index']
#     nhl_full_df['season_stats_id'] = 's - ' + nhl_full_df['season_stats_id'].astype(str)
    
    nhl_full_df = nhl_full_df.drop(['index'], axis = 1, errors = 'ignore')
    # Create unique id for season_year_range
    nhl_full_df['season_year_range_id'] = nhl_full_df.groupby(['season_year_range']).ngroup()
#     nhl_full_df['season_year_range_id'] = 't - ' + nhl_full_df['season_year_range_id'].astype(str)
    
    # Create unique id for season_year_range
#     nhl_full_df['team_id'] = 'y - ' + nhl_full_df['id'].astype(str)

    print('Processed full data')

    return nhl_full_df


###########################################################################################################################################
# NEW CODE BLOCK - Process nhldb tables: teams, time, and season_stats
###########################################################################################################################################

# Teams table
def process_teams_data(nhl_full_df):
    '''
    - Create teams tables
    '''
    teams_df = nhl_full_df[['team_id','team' ]]
    teams_df = teams_df.drop_duplicates(subset = ['team_id'], keep = 'first').reset_index(drop = True)
    print('Processed teams data')

    return teams_df


# Time Table
def process_time_data(nhl_full_df):
    '''
    - Create time table
    '''
    time_df = nhl_full_df[[ 'season_year_range_id', 'season_year_range']]
    time_df = time_df.drop_duplicates(subset = ['season_year_range_id'], keep = 'first').reset_index(drop = True)
    print('Processed time data')

    return time_df


# Season stats table (fact table)
def process_season_stats_data(nhl_full_df):
    '''
    - Create season stats table
    '''
    season_stats_df = nhl_full_df[[
        'season_stats_id',
         'team_id',
         'season_year_range_id',
         'games_played',
         'wins',
         'losses',
         'ot',
         'pts',
         'pt_pctg',
         'goals_per_game',
         'goals_against_per_game',
         'evgga_ratio',
         'power_play_percentage',
         'power_play_goals',
         'power_play_goals_against',
         'power_play_opportunities',
         'penalty_kill_percentage',
         'shots_per_game',
         'shots_allowed',
         'win_score_first',
         'win_opp_score_first',
         'win_lead_first_per',
         'win_lead_second_per',
         'win_outshoot_opp',
         'win_outshot_by_opp',
         'face_offs_taken',
         'face_offs_won',
         'face_offs_lost',
         'face_off_win_percentage',
         'shooting_pctg',
         'save_pctg',
         'failed_shots_per_game',
         'saves_per_game',
         'ties',
         'power_play_efficiency',
         'made_playoffs',
         'adjWins'
    ]]

    print('Processed season stats data')

    return season_stats_df


###########################################################################################################################################
# NEW CODE BLOCK - Insert data into nhldb tables: teams, time, and season_stats
###########################################################################################################################################

# Insert data line by line for conflict conditions
def insert_teams_data(teams_df, conn, cur):
    '''
    - Insert teams data
    '''
    try:
        count = 0
        for index, row in teams_df.iterrows():
            cur.execute(teams_table_insert, list(row))
            
            conn.commit()

            count += 1

            print(' '.join(['Teams data inserted line-by-line into nhldb successfully', str(count)]))

    except ps.Error as e:
        print('\n Error:')
        print(e)

    print(' '.join(['Columns inserted:', str(teams_df.shape[1])]))


# Insert data line by line for conflict conditions
def insert_time_data(time_df, conn, cur):
    '''
    - Insert time data
    '''
    try:
        count = 0
        for index, row in time_df.iterrows():
            cur.execute( time_table_insert, list(row))
            conn.commit()

            count += 1

            print(' '.join(['Time data inserted line-by-line into nhldb successfully', str(count)]))

    except ps.Error as e:
        print('\n Error:')
        print(e)

    print(' '.join(['Columns inserted:', str(time_df.shape[1])]))


# Insert data line by line for conflict conditions
def insert_seasoon_stats_data(season_stats_df, conn, cur):
    '''
    - Insert season stats data
    '''
    try:
        count = 0
        for index, row in season_stats_df.iterrows():
            cur.execute(season_stats_table_insert, list(row))
            conn.commit()

            count += 1

            print(' '.join(['Season stats data inserted line-by-line into nhldb successfully', str(count)]))

    except ps.Error as e:
        print('\n Error:')
        print(e)

    print(' '.join(['Columns inserted:', str(season_stats_df.shape[1])]))


###########################################################################################################################################
# NEW CODE BLOCK - Run etl pipeline
###########################################################################################################################################

def etl():
    '''
    - Run all functions for etl pipeline
    - Select year range for wrangled data in the data directory
    '''
    # Connect to database
    try:

        conn = ps.connect('''
            host=localhost
            dbname=nhldb
            user=postgres
            password=iEchu133
        ''')
        cur = conn.cursor()

        print('Successfully connected to nhldb')

    except ps.Error as e:
        print('\n Database Error:')
        print(e)


    # Input validation
    valid_year_less = date.today().year - 1
    valid_year_greater = 1982

    while True:

        try:
            start_date = int(input("What NHL season do you want the data pull to start from, e.g., 1982? "))
            end_date = int(input(' '.join(["What NHL season do you want the data pull to end with, e.g.,", str(valid_year_less), "? "])))

        except ValueError:
            print(' '.join(["Please select a valid start and end date, e.g., less than or equal to", str(valid_year_less)]))
            print(' '.join(["or greater than or equal to", str(valid_year_greater)]))

        if end_date > valid_year_less or start_date < valid_year_greater:
            print(' '.join(["Please select a valid start and end date, e.g., less than or equal to", str(valid_year_less)]))
            print(' '.join(["or greater than or equal to", str(valid_year_greater)]))

        elif end_date < valid_year_greater or start_date > valid_year_less:
            print(' '.join(["Please select a valid start and end date, e.g., less than or equal to", str(valid_year_less)]))
            print(' '.join(["or greater than or equal to", str(valid_year_greater)]))

        elif end_date < start_date:
            print(' '.join(["Please select a valid start and end date, e.g., less than or equal to", str(valid_year_less)]))
            print(' '.join(["or greater than or equal to", str(valid_year_greater)]))

        else:
            break

    print('Loading...')

    # process season and playoff data
    nhl_full_df = process_data(start_date = start_date, end_date = end_date)

    # process teams table
    teams_df = process_teams_data(nhl_full_df = nhl_full_df)

    # Process time teable
    time_df = process_time_data(nhl_full_df = nhl_full_df)

    # Process season_stats table
    season_stats_df = process_season_stats_data(nhl_full_df = nhl_full_df)

    # Stall to give operator time to scan passed event
    print('Loading...')
    time.sleep(5)

    # Insert teams data
    insert_teams_data(
        teams_df = teams_df, 
        conn = conn, 
        cur = cur
    )

    # Stall to give operator time to scan passed event
    print('Loading...')
    time.sleep(5)

    # Insert time data
    insert_time_data(
        time_df = time_df, 
        conn = conn, 
        cur = cur
    )

    # Stall to give operator time to scan passed event
    print('Loading...')
    time.sleep(5)

    # Insert season_stats data
    insert_seasoon_stats_data(
        season_stats_df = season_stats_df, 
        conn = conn, 
        cur = cur
    )


if __name__ == "__main__":
    etl()
