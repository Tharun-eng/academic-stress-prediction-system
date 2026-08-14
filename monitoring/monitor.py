import os
import sys
import time

# =====================================================
# Add Project Root to Python Path
# =====================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =====================================================
# Import Database Connection
# =====================================================

from database import get_connection


# =====================================================
# Stress Monitor Class
# =====================================================

class StressMonitor:

    def __init__(self):

        self.last_prediction_id = None

        print("\n" + "=" * 70)
        print("ACADEMIC STRESS REAL-TIME MONITOR")
        print("=" * 70)
        print("Monitoring Started Successfully...")
        print("Waiting for New Predictions...\n")

    # =====================================================
    # Get Latest Prediction
    # =====================================================

    def get_latest_prediction(self):

        connection = get_connection()

        if connection is None:
            return None

        try:

            cursor = connection.cursor(dictionary=True)

            cursor.execute("""
                SELECT *
                FROM prediction_history
                ORDER BY id DESC
                LIMIT 1
            """)

            row = cursor.fetchone()

            cursor.close()

            return row

        except Exception as e:

            print("\nDatabase Error :", e)

            return None

        finally:

            if connection.is_connected():
                connection.close()

    # =====================================================
    # Display Prediction
    # =====================================================

    def display_prediction(self, row):

        print("\n" + "=" * 70)
        print("NEW PREDICTION DETECTED")
        print("=" * 70)

        print(f"Prediction ID       : {row['id']}")
        print(f"User ID             : {row['user_id']}")
        print(f"Date & Time         : {row['DateTime']}")

        print("-" * 70)

        print(f"Age                 : {row['Age']}")
        print(f"Gender              : {row['Gender']}")
        print(f"Study Hours         : {row['Study_Hours']}")
        print(f"Screen Time         : {row['Screen_Time']}")
        print(f"Sleep Hours         : {row['Sleep_Hours']}")

        print("-" * 70)

        stress = row["Prediction"]

        print(f"Predicted Stress    : {stress}")

        print("-" * 70)

        if stress == "High":

            print("ALERT STATUS        : HIGH")
            print("ACTION              : Firebase Alert will be sent")

        elif stress == "Medium":

            print("ALERT STATUS        : MEDIUM")
            print("ACTION              : Continue Monitoring")

        else:

            print("ALERT STATUS        : LOW")
            print("ACTION              : Student is Safe")

        print("=" * 70)

    # =====================================================
    # Start Monitoring
    # =====================================================

    def start_monitoring(self):

        while True:

            try:

                latest = self.get_latest_prediction()

                if latest is not None:

                    current_prediction_id = latest["id"]

                    if self.last_prediction_id is None:

                        self.last_prediction_id = current_prediction_id

                        self.display_prediction(latest)

                    elif current_prediction_id != self.last_prediction_id:

                        self.last_prediction_id = current_prediction_id

                        self.display_prediction(latest)

                time.sleep(5)

            except KeyboardInterrupt:

                print("\n")
                print("=" * 70)
                print("REAL-TIME MONITOR STOPPED")
                print("=" * 70)
                break

            except Exception as e:

                print("\nUnexpected Error :", e)

                time.sleep(5)


# =====================================================
# Main Function
# =====================================================

if __name__ == "__main__":

    monitor = StressMonitor()

    monitor.start_monitoring()