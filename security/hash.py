import bcrypt; 

print(bcrypt.hashpw(b'000000', bcrypt.gensalt()).decode())