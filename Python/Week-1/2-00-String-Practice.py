# book: Eric_Matthes_Python_Crash_Course_A_Hands

#practice problem 1 --> name print
print (f'Answer of Question 1:');
name = 'md. nazmul hossain'.title();
print (f'Hello Mr. {name}, would you like to learn some Python today?');


#practice problem 2 --> uppercase. lowercase, titlecase
print (f'\nAnswer of Question 2:');
name = 'mD. naZMul hosSain';
print (f'Default Name: {name}.');
print (f'Your name in uppercase: {name.upper()}.');
print (f'Your name in lowercase: {name.lower()}.');
print (f'Your name in titlecase: {name.title()}.');


#practice problem 3 --> famous quote
print (f'\nAnswer of Question 3:');
message = 'Albert Einstein once said, "A person who never made a mistake never tried anything new."';
print (f'Output: {message}.');


#practice problem 4 --> famous person
print (f'\nAnswer of Question 4:');
famous_person = 'albeRt EinsTein'.title();
message = f'{famous_person} once said, "A person who never made a mistake never tried anything new."';
print (f'Output: {message}.');


#practice problem 5 --> famous quote
print (f'\nAnswer of Question 5:');
message = '    \tMd. Nazmul\n\tHossain               ';
print (f'Default not Strip: {message}.');
print (f'Left Strip: {message.lstrip()}.');
print (f'Right Strip: {message.rstrip()}.');
print (f'Only Strip: {message.strip()}.');
