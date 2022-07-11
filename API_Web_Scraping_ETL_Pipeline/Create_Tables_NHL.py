import psycopg2 as ps
from SQL_Queries_NHL import *

###########################################################################################################################################
# NEW CODE BLOCK - Create nhldb
###########################################################################################################################################

def create_database():
    """
    - Creates and connects to the nhldb
    - Returns the connection and cursor to nhldb
    """

    # connect to default database port: 5432
    conn = ps.connect('''
        host=localhost
        dbname=postgres
        user=postgres
        password=iEchu133
    ''')

    conn.set_session(autocommit = True)
    cur = conn.cursor()

    # create MailChimpdb database with UTF8 encoding
    cur.execute('DROP DATABASE IF EXISTS nhldb;')
    cur.execute("CREATE DATABASE nhldb WITH ENCODING 'utf8' TEMPLATE template0;")

    # close connection to default database
    conn.close()

    # connect to MailChimpdb database
    conn = ps.connect('''
        host=localhost
        dbname=postgres
        user=postgres
        password=iEchu133
    ''')

    cur = conn.cursor()

    return cur, conn


###########################################################################################################################################
# NEW CODE BLOCK - Create tables in nhldb
###########################################################################################################################################


def drop_tables(cur, conn):
    """
    Drops each table using the queries in `drop_table_queries` list
    """
    for query in drop_table_queries:
        cur.execute(query)
        conn.commit()


def create_tables(cur, conn):
    """
    Creates each table using the queries in `create_table_queries` list
    """
    for query in create_table_queries:
        cur.execute(query)
        conn.commit()
    
   ###########################################################################################################################################
# NEW CODE BLOCK - Team names and IDs from NHL API
###########################################################################################################################################

def main():
    """
    - Drops (if exists) and creates the nhldb database
    - Establishes connection with the nhldb database and gets cursor to it
    - Drops all the tables
    - Creates all tables needed
    - Finally, closes the connection
    """

    try:
        cur, conn = create_database()

        drop_tables(cur, conn)
        
        create_tables(cur, conn)
        print('Tables have been created: teams, time, and season_stats')

        conn.close()

    except ps.Error as e:
        print('\n Error:')
        print(e)


if __name__ == "__main__":
    main()
