from pprint import pprint

def rule_1(jug_1, jug_2):
    """Fill 'jug_1' with 4."""
    jug_1 = 4
    return jug_1, jug_2

def rule_2(jug_1, jug_2):
    """"Fill 'jug_2' with 3."""
    jug_2 = 3
    return jug_1, jug_2

def rule_3(jug_1, jug_2):
    """Empty 'jug_1'."""
    jug_1 = 0
    return jug_1, jug_2

def rule_4(jug_1, jug_2):
    """Empty 'jug_2'."""
    jug_2 = 0
    return jug_1, jug_2

def rule_5(jug_1, jug_2):
    """Pour 'jug_2' to 'jug_1'."""
    total = jug_1 + jug_2
    if (total > 4 ):
        rem = total - 4
        jug_1 = 4
        jug_2 = rem
    else:
        jug_1 = total
        jug_2 = 0
    return jug_1, jug_2

def rule_6(jug_1, jug_2):
    """Pour 'jug_1' to 'jug_2'."""
    total = jug_1 + jug_2
    if (total > 3 ):
        rem = total - 3
        jug_2 = 3
        jug_1 = rem
    else:
        jug_2 = total
        jug_1 = 0
    return jug_1, jug_2

def path_to_goal(rules, visited, stack, rule_no=0, jug_1=0, jug_2=0):
    pair = (jug_1, jug_2)
    stack.append([pair, "rule - " + str(rule_no)])
    if pair == (2,0):
        return True
    elif pair == (2,1):
        return True
    elif pair == (2,2):
        return True
    elif pair == (2,3):
        return True
    if pair in visited:
        return
    visited.append(pair)
    counter = 1
    for rule in rules:
        new_jug_1, new_jug_2 = rule(jug_1, jug_2)
        new_pair = (new_jug_1, new_jug_2)
        if new_pair not in visited:
            complete = path_to_goal(rules, visited, stack, counter, new_jug_1, new_jug_2)
            if complete:
                return True
            stack = stack[:-1]
        counter += 1

def main():
    rules = []
    visited = []
    stack = []
    rules.append(rule_1)
    rules.append(rule_2)
    rules.append(rule_3)
    rules.append(rule_4)
    rules.append(rule_5)
    rules.append(rule_6)
    path_to_goal(rules, visited, stack)
    pprint(stack)


if __name__ == '__main__':
    main()
