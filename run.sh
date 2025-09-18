
root_dir=/checkpoint/ocp/misko/traces/no_graph_gen/
natoms="1000 2000 4000 8000 16000 32000"
ngpus="1 2 4 8"
compiles="False True"

for natom in $natoms; do 
for ngpu in $ngpus; do 
for compile in $compiles; do 
if [ `expr $natom / $ngpu` -gt "4000" ]; then
	continue
fi

d="${root_dir}/natoms${natom}_ngpus${ngpu}_compile${compile}_main"
echo $d
mkdir -p ${d}
cat fairchem/configs/uma/benchmark/uma-speed.yaml | sed "s/_NATOMS_/${natom}/g" | sed "s/_RANKS_/${ngpu}/g" | sed "s/_COMPILE_/${compile}/g" > ${d}/config.yaml
fairchem -c ${d}/config.yaml +run_dir_root=${d}/ > ${d}/log.txt 2>&1 

done
done
done
