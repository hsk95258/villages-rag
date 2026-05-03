import psycopg2, os
from dotenv import load_dotenv
load_dotenv()
conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cursor = conn.cursor()
cursor.execute("SELECT tablename FROM pg_tables WHERE schemaname='public'")
print(cursor.fetchall())
conn.close()