#book: Eric_Matthes_Python_Crash_Course_A_Hands

#assign message to variable with single quote
message = 'Hello Nazmul..!';
print (message);

#reassign message to variable with double quote
message = "I am from Bangladesh.";
print(message);

#write a message in random case
#title method convert this to Md. Nazmul Hossain format
message = 'mD. maZmul hoSsaiN';
print(message.title());

#covert upper message into lowercsae
message = message.lower();
print(message);

#covert upper message into uppercase
message = message.upper();
print(message);


#Using Variables in Strings
fName = 'md';
mName = 'nazmul';
lName = 'hossain';
message = f"{fName} {mName} {lName}.";
print("Hello, Mr.", message.title());


#Adding Whitespace to Strings with Tabs or Newlines
print("Hello World:\n\tI am Md. Nazmul Hossain\n\tFrom Green University of Bangladesh.");


#Stripping Whitespace remove left/right Whitespace
message = "           Hello World                 ";
print(message);
print(f'left trim !--{message.lstrip()}--!');
print(f'Right trim !--{message.rstrip()}--!');
message = message.strip();
print(f'!--{message}--!');
