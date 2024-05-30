for i in {0..20}
do
	nohup python random_generator.py $i 2 > $i.out &
done
