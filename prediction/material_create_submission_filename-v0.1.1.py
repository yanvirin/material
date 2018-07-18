#!/usr/bin/python
# -*-coding:Utf-8 -*
'''
    File name: material_create_submission_filename-v0.1.1.py
    Author: Sarra Chouder
    Contact Email: material_poc@nist.gov
    Date created: 04/20/2017
    Version: 0.1.1
'''
import os,sys
import re
import time
import argparse
import shutil

def renameSub():
	parser = argparse.ArgumentParser(description='MATERIAL Renaming tool: renames a submission to meet the Eval Plan requirements')
	parser.add_argument("--submission_file", required=True, help="Path of the submission you want to rename")
	parser.add_argument("--team", required=True, help="Choose a team", choices=['FLAIR','QuickSTIR','SARAL','SCRIPTS'])
	parser.add_argument("--task", required=True, help="Choose a task", choices=['CLIR','DomainID','E2E'])
	parser.add_argument("--sub_type", required=True, help="Choose a submission type", choices=['primary','contrastive'])
	parser.add_argument("--set", required=True, help="Choose a queryset (QUERY1: 300, QUERY2: 400, QUERY1QUERY2: 700) OR a domainset (D12: GOV-LIF [1A/1B], D1234: GOV-LIF-BUS-LAW [1A] GOV-LIF-HEA-MIL [1B])", choices=['QUERY1','QUERY2','QUERY1QUERY2','D12','D1234'])
	parser.add_argument("--lang", required=True, help="Choose a language", choices=['1A','1B'])
	parser.add_argument("--dataset", required=True, help="Choose a dataset", choices=['DEV','ANALYSIS1','DEVANALYSIS1','EVAL1EVAL2'])
	parser.add_argument("--outpath", help="Path where you want the renamed submission to be copied. By default, it will be copied in the current working directory.")
	args = parser.parse_args()

	if (args.task=="CLIR" and args.set.startswith('D')) or (args.task=="DomainID" and args.set.startswith('QUERY')) or (args.task=="E2E" and args.set.startswith('D')):
		print("--ERROR. You can not choose that set with that task")
		print("Please, retry!")
		sys.exit(1)

	if (args.task=="DomainID" and args.set=="D12"):
		args.set="GOV-LIF"
	elif (args.task=="DomainID" and args.lang=="1A" and args.set=="D1234"):
		args.set="GOV-LIF-BUS-LAW"
	elif (args.task=="DomainID" and args.lang=="1B" and args.set=="D1234"):
		args.set="GOV-LIF-HEA-MIL"
	
	head, tail = os.path.split(args.submission_file)
	filename, file_ext = os.path.splitext(tail)
	fn = re.sub('[^0-9a-zA-Z]', '-', filename)
	sub_name = args.team+'_'+args.task+'-'+args.sub_type+'-unconstrained-'+args.set+'-'+fn+'_BASE-'+args.lang+'-'+args.dataset+'_'+time.strftime("%Y%m%d")+'_'+time.strftime("%H%M%S")+file_ext
	
	print("\nYour submission will be renamed \"%s\""%sub_name)

	if args.outpath is not None:
		print("\nSubmission copied and renamed in the outpath you provided.")
		shutil.copy(args.submission_file, args.outpath+'/'+sub_name)
	else:
		print("\nSubmission copied and renamed in your current working directory.")
		shutil.copy(args.submission_file, sub_name)
	print("\n**Done**")

if __name__ == '__main__':
	renameSub()
