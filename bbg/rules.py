rules = """
Instructions:
ED: Euclidean Distance
NED: Normalized Euclidean Distance (0~1)
LR: Length Ratio (0~1)
TA: Turning Angle (0-180)
R_: Rank of ...

[1] R_ED1 <= 2
[2] ED < 3*30
[3] if ED>2*30 then LR<0.999
[4] if ED>1*30 then TA<=180
[5] if ED>1*30 then LR<0.99
[6] if ED>5*30 then TA<60
[7] LR<1.1
"""