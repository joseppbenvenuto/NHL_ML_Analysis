from NHL_API_Web_Scraper import get_nhl_data
from ETL_NHL import etl

import sys


def main():
    '''
    - Run both API & web scrapr for season and playoff data and ETL pipeline
    '''
###########################################################################################################################################
# NEW BLOCK - Module 1 API & web scrapr for season and playoff data
###########################################################################################################################################

    print('''
    Module 1:
    Wrangles data from the NHL API https://www.kevinsidwar.com/iot/2017/7/1/the-undocumented-nhl-stats-api (season data) 
    and scrapes data from https://www.hockey-reference.com (playoff data).
    ''')

    yesChoice = ['yes','y']
    noChoice = ['no','n']

    input_1 = input("Would you like to run Module 1? ['yes','y'] or ['no','n'] ")
    input_1 = input_1.lower()

    if input_1 in yesChoice:
        try:
            '''
            - Warnagle and store NHL season and playoff data to data storage directories via CSV
            '''
            get_nhl_data()
            input("Press enter to continue to Module 2 or Ctrl C to end the program.")
        except Exception as e:
            print(e)
            input("Please check the noted error above and decide to press enter and continue to Module 2 or Ctrl C to end the program.")

    elif input_1 in noChoice:
        print('You have skipped Module 1.')
        pass

    else:
        input('You have entered an incorrect response, please press enter to end the program.')
        sys.exit()


###########################################################################################################################################
# NEW BLOCK - ETL Module
###########################################################################################################################################

    print('''
    Module 2:
    Runs the ETL pipline to store the NHL season and playoff data in PostgreSQL database (nhldb).
    ''')

    string = "Would you like to run the ETL Module? ['yes','y'] or ['no','n']"
    print(string)
    input_2 = input()
    input_2 = input_2.lower()

    if input_2 in yesChoice:
        try:
            '''
            - Run mc report ETL to nhldb
            '''
            etl()
            input("ETL Module complete, please press enter or Ctrl C to end the program.")
        except Exception as e:
            print(e)
            input("Please check the noted error above.")

    elif input_2 in noChoice:
        print('You have skipped the ETL Module.')
        print('Program complete.')
        pass

    else:
        input('You have entered an incorrect response, please press enter to end the program.')
        sys.exit()
        
        
if __name__ == "__main__":
    main()