import pandas as pd
from datetime import datetime
from database import get_connection

# Read CSV
df = pd.read_csv("history/prediction_history.csv")

connection = get_connection()
cursor = connection.cursor()

count = 0

for _, row in df.iterrows():

    # Convert DD-MM-YYYY to YYYY-MM-DD
    mysql_datetime = datetime.strptime(
        str(row["DateTime"]),
        "%d-%m-%Y %H:%M:%S"
    ).strftime("%Y-%m-%d %H:%M:%S")

    sql = """
    INSERT INTO prediction_history
    (DateTime, Age, Study_Hours, Screen_Time, Sleep_Hours, Prediction)
    VALUES (%s, %s, %s, %s, %s, %s)
    """

    values = (
        mysql_datetime,
        int(row["Age"]),
        int(row["Study_Hours"]),
        int(row["Screen_Time"]),
        int(row["Sleep_Hours"]),
        row["Prediction"]
    )

    cursor.execute(sql, values)
    count += 1

connection.commit()

print(f"\n✅ {count} Records Imported Successfully!")

cursor.close()
connection.close()