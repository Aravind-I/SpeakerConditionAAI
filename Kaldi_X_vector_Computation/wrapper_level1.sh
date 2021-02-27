 for name in `cat Subjects.txt`; 
	do echo "current Subject "$name;
	./wrapper_level2.sh $name
 done
