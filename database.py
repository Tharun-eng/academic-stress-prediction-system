import os

from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

print("database.py loaded")


# ==========================================
# DATABASE CONNECTION
# ==========================================

def get_connection():

    print("Inside get_connection()")

    try:

        connection = mysql.connector.connect(

            host=os.getenv(
                "MYSQL_HOST",
                "127.0.0.1"
            ),

            port=int(
                os.getenv(
                    "MYSQL_PORT",
                    "3306"
                )
            ),

            user=os.getenv(
                "MYSQL_USER",
                "root"
            ),

            password=os.getenv(
                "MYSQL_PASSWORD"
            ),

            database=os.getenv(
                "MYSQL_DATABASE",
                "academic_stress_db"
            ),

            connection_timeout=10,

            use_pure=True
        )

        if connection.is_connected():

            print("✅ Connected Successfully!")

            return connection

        print(
            "❌ Connection object created, "
            "but not connected."
        )

        return None

    except Error as e:

        print("❌ MySQL Error:")
        print(e)

        return None

    except Exception as e:

        print("❌ Unexpected Error:")
        print(e)

        return None


# ==========================================
# SAVE PREDICTION
# ==========================================

def save_prediction(data):

    connection = get_connection()

    if connection is None:

        print(
            "❌ Database connection failed. "
            "Prediction was not saved."
        )

        return False

    cursor = None

    try:

        cursor = connection.cursor()

        sql = """
        INSERT INTO prediction_history
        (
            DateTime,
            Age,
            Study_Hours,
            Screen_Time,
            Sleep_Hours,
            Prediction
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        """

        values = (

            data["DateTime"],

            data["Age"],

            data["Study_Hours"],

            data["Screen_Time"],

            data["Sleep_Hours"],

            data["Prediction"]

        )

        cursor.execute(
            sql,
            values
        )

        connection.commit()

        print(
            "✅ Prediction saved successfully!"
        )

        return True

    except Error as e:

        print(
            "❌ Save Prediction Error:"
        )

        print(e)

        if connection.is_connected():

            connection.rollback()

        return False

    except Exception as e:

        print(
            "❌ Unexpected Save Error:"
        )

        print(e)

        if connection.is_connected():

            connection.rollback()

        return False

    finally:

        if cursor is not None:

            cursor.close()

        if connection is not None:

            connection.close()


# ==========================================
# TEST DATABASE CONNECTION
# ==========================================

if __name__ == "__main__":

    print(
        "\n========== DATABASE TEST =========="
    )

    connection = get_connection()

    if connection is not None:

        print(
            "✅ Database test successful!"
        )

        connection.close()

    else:

        print(
            "❌ Database test failed!"
        )

    print(
        "===================================\n"
    )