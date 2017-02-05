to_translate = raw_input("Please enter phrase you want to translate to whale: ")

vowels = ["a", "i", "o", "u", "e", "A", "E", "I", "O", "U"]


whale_string = ""

for c in to_translate:
    if c in vowels:
        extended = ""
        for i in range(5):
            extended  = extended + c
        whale_string = whale_string + extended
    else:
        whale_string = whale_string + c

print("This is your translastion!")
print(whale_string)
