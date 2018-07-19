#!/usr/bin/python
# -*-coding:Utf-8 -*
'''
    File name: material_create_submission_filename-v0.1.3.py
    Author: Sarra Chouder
    Contact Email: material_poc@nist.gov
    Date created: 04/20/2017
    Version: 0.1.3
'''
import os,sys,errno
import re
import time
import argparse
import shutil

def renameSub():
	parser = argparse.ArgumentParser(description='MATERIAL Renaming tool: renames a submission to meet the Eval Plan requirements')
	parser.add_argument("--submission_file", required=True, help="Path of the submission you want to rename")
	parser.add_argument("--team", required=True, help="Choose a team", choices=['FLAIR','QuickSTIR','SARAL','SCRIPTS'])
	parser.add_argument("--task", required=True, help="Choose a task", choices=['CLIR','DomainID','E2E','LanguageID'])
	parser.add_argument("--sub_type", required=True, help="Choose a submission type", choices=['primary','contrastive'])
	parser.add_argument("--set", help="Choose a queryset (QUERY1: 300, QUERY2: 400, QUERY1QUERY2: 700, QUERY2QUERY3: 1000) OR a domainset (DX: GOV-LIF [1A/1B], DXY: GOV-LIF-BUS-LAW [1A] GOV-LIF-HEA-MIL [1B], DXYZ: GOV-LIF-BUS-LAW-SPO [1A] GOV-LIF-HEA-MIL-SPO [1B])", choices=['QUERY1','QUERY2','QUERY1QUERY2','QUERY2QUERY3','DX','DXY','DXYZ'])
	parser.add_argument("--lang", required=True, help="Choose a language", choices=['1A','1B'])
	parser.add_argument("--dataset", required=True, help="Choose a dataset", choices=['DEV','ANALYSIS1','DEVANALYSIS1','EVAL1EVAL2','EVAL1EVAL2EVAL3','ANALYSIS1ANALYSIS2'])
	parser.add_argument("--outpath", help="Path where you want the renamed submission to be copied. By default, it will be copied in the current working directory.")
	args = parser.parse_args()

	#if args.task=="CLIR" or args.task=="E2E" or args.task=="DomainID":
	#	parser.error("argument --set is required")

	if ((args.task=="CLIR" or args.task=="E2E") and args.set.startswith('D')) or (args.task=="DomainID" and args.set.startswith('QUERY')):
		print("--ERROR. You can not choose that set with that task")
		print("Please, retry!")
		sys.exit(1)

	if (args.task=="DomainID" and args.set=="DX"):
		args.set="GOV-LIF"
	elif (args.task=="DomainID" and args.lang=="1A" and args.set=="DXY"):
		args.set="GOV-LIF-BUS-LAW"
	elif (args.task=="DomainID" and args.lang=="1B" and args.set=="DXY"):
		args.set="GOV-LIF-HEA-MIL"
	elif (args.task=="DomainID" and args.lang=="1A" and args.set=="DXYZ"):
		args.set="GOV-LIF-BUS-LAW-SPO"
	elif (args.task=="DomainID" and args.lang=="1B" and args.set=="DXYZ"):
		args.set="GOV-LIF-HEA-MIL-SPO"
	
	head, tail = os.path.split(args.submission_file)
	filename, file_ext = os.path.splitext(tail)
	fn = re.sub('[^0-9a-zA-Z]', '-', filename)
	if args.task=="LanguageID":
		sub_name = args.team+'_'+args.task+'-'+args.sub_type+'-unconstrained-'+fn+'_BASE-'+args.lang+'-'+args.dataset+'_'+time.strftime("%Y%m%d")+'_'+time.strftime("%H%M%S")+file_ext
	else:
		sub_name = args.team+'_'+args.task+'-'+args.sub_type+'-unconstrained-'+args.set+'-'+fn+'_BASE-'+args.lang+'-'+args.dataset+'_'+time.strftime("%Y%m%d")+'_'+time.strftime("%H%M%S")+file_ext
	
	print("\nYour submission will be renamed \"%s\""%sub_name)

	if args.outpath is not None:
		if not os.path.exists(args.outpath):
			os.makedirs(args.outpath)
			print("\nDirectory '%s' created."%args.outpath)
		print("\nSubmission copied and renamed in the outpath you provided.")
		shutil.copy(args.submission_file, args.outpath+'/'+sub_name)
	else:
		print("\nSubmission copied and renamed in your current working directory.")
		shutil.copy(args.submission_file, sub_name)
	print("\n**Done**")

if __name__ == '__main__':
	renameSub()
