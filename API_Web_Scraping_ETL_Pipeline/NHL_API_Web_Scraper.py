# Import libraries
from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
import numpy as np
import time
from datetime import date

import warnings
warnings.filterwarnings('ignore')


###########################################################################################################################################
# NEW CODE BLOCK - Team names and IDs from NHL API
###########################################################################################################################################

# User Guide : https://www.kevinsidwar.com/iot/2017/7/1/the-undocumented-nhl-stats-api
def nhl_id(team_id):
    '''
    - Get individual team names and ids by index number 0-31
    '''
    url = "https://statsapi.web.nhl.com/api/v1/teams"

    response = requests.get(url)
    data = response.json()
    data = data['teams']
    data = data[team_id]
    data_id = data['id']
    data_name = data['name']

    id_dict = [{"id" : data_id, "name" : data_name}]

    return id_dict

# Function to retrieve NHL team ids from NHL API
def nhl_id_all(nhl_id):
    '''
    - Returns all unique ids for the 32 NHL teams, this will allows us to pull data from NHL API
    '''
    index = 0
    request = 0
    max_request = 34
    id_list = []

    while request < max_request:
        try:
            id = nhl_id(index)
            id_list = id_list + id
            index += 1
            request += 1

        except IndexError:
            df = pd.DataFrame(id_list)

            return df


###########################################################################################################################################
# NEW CODE BLOCK - Team stats per team per season from NHL API
###########################################################################################################################################

# Team stats per team per season
def nhl_stats(id_num, year_range):
    '''
    - Queries teams stats by id for a given year/season e.g 19831984 = 1983 - 1984
    '''
    url = "http://statsapi.web.nhl.com/api/v1/teams/{}?expand=team.stats&season={}".format(id_num, year_range)

    response = requests.get(url)
    data = response.json()
    data = data['teams']
    data = data[0]
    data = data['teamStats']
    data = data[0]
    data = data['splits']
    data = data[0]
    data = data['stat']

    return data


# All team stats per team per season
def nhl_season_stats(year, nhl_id_all, nhl_id, nhl_stats):
    '''
    - Get season stats per chosen season for all teams
    '''
    team_id = nhl_id_all(nhl_id = nhl_id)

    id = -1
    request = -1
    max_request = 31
    stats_list = []

    while request <= max_request:
        try:
            id += 1
            request += 1
            id_num = team_id.loc[id, 'id']
            id_name = team_id.loc[id, 'name']
            stats = nhl_stats(id_num = id_num, year_range = year)
            stats['id'] = id_num
            stats['name'] = id_name
            stats['year_range'] = year
            stats_list = stats_list + [stats]
            time.sleep(1)

        except:
            continue

    return stats_list


###########################################################################################################################################
# NEW CODE BLOCK - Multiple season stats per team from NHL API
###########################################################################################################################################

# Function to loop through multiple seasons of team stats
def nhl_table_multiple_seasons(year1_range, max_year, nhl_id_all, nhl_id, nhl_stats, nhl_season_stats):
    '''
    - To loop through multiple seasons of team stats
    '''
    year1 = int(year1_range[0:4])
    year2 = int(year1_range[5:9])
    request = -1
    max_request = (max_year - year2) - 1
    index = -1
    full_stats_list = []

    while request < max_request:
        index += 1
        year1 += 1
        year2 += 1
        request += 1
        year_range = '{}{}'.format(year1, year2)
        print('Season: ', year1, year2, index, year_range)
        
        stats = nhl_season_stats(
            year = year_range, 
            nhl_id_all = nhl_id_all,
            nhl_id = nhl_id, 
            nhl_stats = nhl_stats
        )
        
        full_stats_list = full_stats_list + stats
        time.sleep(1)

    # Convert list to data frame
    nhl_df = pd.DataFrame(full_stats_list)

    # Clean out speacial characters in teams column
    nhl_df['name'] = nhl_df['name'].str.replace(r'[^\x00-\x7F]+', '')

    return nhl_df


###########################################################################################################################################
# NEW CODE BLOCK - Web scrape playoff instances per team per season from NHL API
###########################################################################################################################################

# https://www.hockey-reference.com/
def get_payoff_teams(year1, year2):
    '''
    - Get teams that were in the playoffs for that given season
    '''
    playoff_teams_list = []
    # Loop through years for playoff teams
    for year in range(year1, year2 + 1):
        url = 'https://www.hockey-reference.com/playoffs/NHL_{}.html'.format(year)

        print('year range: ' + str(year - 1) + str(year))

        response = requests.get(url)
        soup = bs( response.text, 'html.parser')

        # Scrape live playoff team data
        team = soup.select('.table_wrapper > .table_container > table > tbody > tr > .left')
        team = [x.get_text().strip() for x in team]
        teams = team[0:16]
        print('number of playoff teams: ' + str(len(teams)))
        
        full_year_range = [str(year - 1) + str(year)] * 16
        playoff_teams_dict = {
            'playoff_teams': teams,
            'playoff_year_range': full_year_range
        }
        
        try:
            playoff_teams_df = pd.DataFrame(playoff_teams_dict)

        except:
            continue

        playoff_teams_list.append(playoff_teams_df)

        time.sleep(1)

    df = pd.concat(playoff_teams_list)

    df['id'] = np.where(
    df['playoff_teams'] == 'Anaheim Ducks', 24,
    np.where(
        df['playoff_teams'] == 'Atlanta Thrashers', 100,
        np.where(
            df['playoff_teams'] == 'Boston Bruins', 6,
             np.where(
                 df['playoff_teams'] == 'Buffalo Sabres', 7,
                 np.where(
                     df['playoff_teams'] == 'Calgary Flames', 20,
                     np.where(
                         df['playoff_teams'] == 'Carolina Hurricanes', 12,
                         np.where(
                             df['playoff_teams'] == 'Chicago Black Hawks', 16,
                             np.where(
                                 df['playoff_teams'] == 'Chicago Blackhawks', 16,
                                 np.where(
                                     df['playoff_teams'] == 'Colorado Avalanche', 21,
                                     np.where(
                                         df['playoff_teams'] == 'Columbus Blue Jackets', 29,
                                          np.where(
                                              df['playoff_teams'] == 'Dallas Stars', 25,
                                              np.where(
                                                  df['playoff_teams'] == 'Detroit Red Wings', 17,
                                                  np.where(
                                                      df['playoff_teams'] == 'Edmonton Oilers', 22,
                                                      np.where(
                                                          df['playoff_teams'] == 'Florida Panthers', 13,
                                                          np.where(
                                                              df['playoff_teams'] == 'Hartford Whalers', 100,
                                                              np.where(
                                                                  df['playoff_teams'] == 'Los Angeles Kings', 26,
                                                                  np.where(
                                                                      df['playoff_teams'] == 'Mighty Ducks of Anaheim', 24,
                                                                      np.where(
                                                                          df['playoff_teams'] == 'Minnesota North Stars', 100,
                                                                          np.where(
                                                                              df['playoff_teams'] == 'Minnesota Wild', 30,
                                                                              np.where(
                                                                                  df['playoff_teams'] == 'Montreal Canadiens', 8,
                                                                                  np.where(
                                                                                      df['playoff_teams'] == 'Nashville Predators', 18,
                                                                                      np.where(
                                                                                          ['playoff_teams'] == 'New York Islanders', 2,
                                                                                          np.where(
                                                                                              df['playoff_teams'] == 'New York Rangers', 3,
                                                                                              np.where(
                                                                                                  df['playoff_teams'] == 'Ottawa Senators', 9,
                                                                                                  np.where(
                                                                                                      df['playoff_teams'] == 'Philadelphia Flyers', 4,
                                                                                                      np.where(
                                                                                                          df['playoff_teams'] == 'Phoenix Coyotes', 53,
                                                                                                          np.where(
                                                                                                              df['playoff_teams'] == 'Pittsburgh Penguins', 5,
                                                                                                               np.where(
                                                                                                                   df['playoff_teams'] == 'Quebec Nordiques', 100,
                                                                                                                   np.where(
                                                                                                                       df['playoff_teams'] == 'San Jose Sharks', 28,
                                                                                                                       np.where(
                                                                                                                           df['playoff_teams'] == 'St. Louis Blues', 19,
                                                                                                                           np.where(
                                                                                                                               df['playoff_teams'] == 'Tampa Bay Lightning', 14,
                                                                                                                               np.where(
                                                                                                                                   df['playoff_teams'] == 'Toronto Maple Leafs', 10,
                                                                                                                                   np.where(
                                                                                                                                       df['playoff_teams'] == 'Vancouver Canucks', 23,
                                                                                                                                       np.where(
                                                                                                                                           df['playoff_teams'] == 'Vegas Golden Knights', 54,
                                                                                                                                           np.where(
                                                                                                                                               df['playoff_teams'] == 'Washington Capitals', 15,
                                                                                                                                               np.where(
                                                                                                                                                   df['playoff_teams'] == 'Winnipeg Jets', 52,
                                                                                                                                                   np.where(
                                                                                                                                                       df['playoff_teams'] == 'New Jersey Devils', 1, 100 
                                                                                                                                                       )
                                                                                                                                                    )
                                                                                                                                                )
                                                                                                                                            )
                                                                                                                                        )
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            )
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )

    return df


###########################################################################################################################################
# NEW CODE BLOCK - Run all functions to get season stats from 1983 to 2021 in addition too teams who qualified for the playoffs
###########################################################################################################################################


def get_nhl_data():
    '''
    - Select year range for data wrangling season and playoff data
    - Run all functions
    - Store warngled data to data directory
    '''
    # Input validation
    valid_year_less = date.today().year - 1
    valid_year_greater = 1982

    while True:

        try:
            start_date = int(input("What NHL season do you want the data pull to start from, e.g., 1982? "))
            end_date = int(input( "What NHL season do you want the data pull to end with, e.g., " + str(valid_year_less) + "? "))

        except ValueError:
            print("Please select a valid start and end date, e.g., less than or equal to " + str(valid_year_less))
            print("or greater than or equal to " + str(valid_year_greater) + ".")

        if end_date > valid_year_less or start_date < valid_year_greater:
            print("Please select a valid start and end date, e.g., less than or equal to " + str(valid_year_less))
            print("or greater than or equal to " + str(valid_year_greater) + ".")

        elif end_date < valid_year_greater or start_date > valid_year_less:
            print("Please select a valid start and end date, e.g., less than or equal to " + str(valid_year_less))
            print("or greater than or equal to " + str(valid_year_greater) + ".")

        elif end_date < start_date:
            print("Please select a valid start and end date, e.g., less than or equal to " + str(valid_year_less))
            print("or greater than or equal to " + str(valid_year_greater) + ".")

        else:
            break

    print('getting playoff data')

    # scrape teams who qualified for playoffs per season data
    nhl_playoff_df = get_payoff_teams(year1 = start_date,year2 = end_date)

    # Export data frame to project directory
    file = 'Data/' + str(start_date) + '_' + str(end_date) + '_' + '_NHL_Playoff_Data.csv'
    nhl_playoff_df.to_csv(
        file,
        sep = ',',
        encoding = 'utf-8',
        index = False
    )

    print('getting season data')

    # Run all functions and create season stats data
    start_range = str(start_date) + "," + str(start_date + 1)

    nhl_df = nhl_table_multiple_seasons(
        year1_range = start_range,
        max_year = end_date,
        nhl_id_all = nhl_id_all,
        nhl_id = nhl_id,
        nhl_stats = nhl_stats,
        nhl_season_stats = nhl_season_stats
    )


    # Export data frame to project directory
    file = 'Data/' + str(start_date) + '_' + str(end_date) + '_' + '_NHL_Season_Data.csv'
    nhl_df.to_csv(
        file,
        sep = ',',
        encoding = 'utf-8',
        index = False
    )


if __name__ == "__main__":
    get_nhl_data()
