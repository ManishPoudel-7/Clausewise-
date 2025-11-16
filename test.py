from pymongo import MongoClient
import certifi

client = MongoClient("mongodb+srv://manishpoudel026_db_user:ktFutELomsoiv33u@clausewise.wtmeijf.mongodb.net/?retryWrites=true&w=majority&appName=ClauseWise",
                     tlsCAFile=certifi.where())

print(client.list_database_names())
