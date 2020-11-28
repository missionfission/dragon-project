"""
fn(area constraint, graph)
analyze graph to get a good initial value of hw description
config.create()
gen_systl_fn()
write as 0_hw.yaml
"""

"""
fn(config)
generate register files and scratchpad config from HW config
return full_config
"""


"""
gen_systl_fn(graph, config)
systolic array config from mapping efficiency
mapping_efficiency total -> (total time of execution for this systolic array config)
bigger conv layers should have better mapping efficiency
first conv layer -> time1
second conv layer -> time2
third layer -> time3
total time -> minimize -> time1 + time2 + time3
select one which minimizes this sum
try some configs and select the best one
return full_config
"""
