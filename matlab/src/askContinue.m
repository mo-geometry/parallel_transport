function askContinue(str1)

fprintf('%s \n',str1)

m = input('(y/n) \n','s');
if m=='n',
    error('Sequence terminated.');  
end