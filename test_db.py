print("Program Started")

from database import get_connection

print("Imported Successfully")

print("Trying Connection...")

connection = get_connection()

if connection is not None:

    print("Returned from function")

    print("Connection Status:", connection.is_connected())

    connection.close()

    print("✅ Connection Closed")

else:

    print("❌ get_connection() returned None")