def create_function_list(rg):
    func_list = []
    for i in range(rg):
        def fc_add(x):
            out = x+i
            print(i, out)
            return out

        func_list.append(fc_add)
    return func_list

fc_ls = create_function_list(4)

a = 0

for fc in fc_ls:
    a = fc(a)

