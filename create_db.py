# create_db.py

import sqlite3

connection = sqlite3.connect('comments.db')
cursor = connection.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS Shows
              (User TEXT, Comment TEXT)''')

connection.commit()
connection.close()
