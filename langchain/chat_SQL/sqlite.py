import sqlite3


#connect to sqlite
connection=sqlite3.connect("student.db")

#create a cursor object to insert record, creat table
cusror = connection.cursor()

# create the table
table_info= """
create table STUDENT(NAME VARCHAR(25), CLASS VARCHAR(25), 
SECTION VARCHAR(25), MARKS INT)
"""

cusror.execute(table_info)

# Insert some records
cusror.execute('''Insert Into STUDENT values('Mithu', 'Data Science', 'A', 90)''')
cusror.execute('''Insert Into STUDENT values('Jhon', 'Data Science', 'B', 80)''')
cusror.execute('''Insert Into STUDENT values('Doe', 'Developer', 'A', 85)''')
cusror.execute('''Insert Into STUDENT values('Jony', 'Data Science', 'A', 85)''')
cusror.execute('''Insert Into STUDENT values('Clark', 'Data Science', 'B', 90)''')
cusror.execute('''Insert Into STUDENT values('Mical', 'Developer', 'A', 95)''')



#Display all the records
print("The Inserted record are:")
data = cusror.execute('''Select * from STUDENT''')

for row in data:
    print(row)

# Commit your changes
connection.commit()
connection.close()