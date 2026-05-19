import pickle, os
for p in ['checkpoints/train01_slots_real4.pkl','checkpoints/train01_slots.pkl','checkpoints/train01_slots_real3.pkl','checkpoints/train01_slots_real2.pkl']:
    if os.path.exists(p):
        try:
            data = pickle.load(open(p,'rb'))
            print('\nFILE:',p)
            print(' top keys:', list(data.keys()))
            for split in data:
                print(' split',split,'count',len(data[split]))
                for k,v in list(data[split].items())[:5]:
                    print('  ',k,'shape',getattr(v,'shape',None))
        except Exception as e:
            print(p,'load error',e)
    else:
        print('\nFILE NOT FOUND:',p)

print('\nCHECKPOINTS DIR LISTING:')
print('\n'.join(sorted(os.listdir('checkpoints'))))
