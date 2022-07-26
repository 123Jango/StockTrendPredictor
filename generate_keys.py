import pickle
from pathlib import Path

import streamlit_authenticator as stauth
names =["Sayan Chakraborty","KIIT"]
usernames = ["Scarab","Dolphin"]
passwords = ["Sc@123","DOL@567"]

#streamlit authenticator uses bcrypt for password hashing which is secured as very secure algorithm
hashed_passwords = stauth.Hasher(passwords).generate()

file_path = Path(__file__).parent/ "hashed_PW.pkl"
with file_path.open('wb') as file:
    pickle.dump(hashed_passwords, file)