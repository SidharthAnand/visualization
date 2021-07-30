f=open('E:/work/week9/Poses/Before 6-22/standing_10_8x8_sensor.txt','r')

res = f.read()
res = res.replace("[[b'{","{")
res = res.replace("}']]","}")
res = res.replace("', b'",'\n')
res = res.replace("'], [b'",'\n')

s=open('E:/work/week9/standing_10_8x8_t.txt','a+')
s=s.write(res)