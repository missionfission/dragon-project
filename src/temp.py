# if (read_bw_ll < self.mem_read_bw[self.mle - 1] and write_bw_ll < self.mem_write_bw[self.mle - 1]):
# elif (
#     read_bw_ll < self.mem_read_bw[self.mle - 1]
#     and write_bw_ll > self.mem_write_bw[self.mle - 1]
# ):
#     step_cycles = write_bw_ll / self.mem_write_bw[self.mle - 1]
# elif (
#     read_bw_ll > self.mem_read_bw[self.mle - 1]
#     and write_bw_ll < self.mem_write_bw[self.mle - 1]
# ):
#     step_cycles = read_bw_ll / self.mem_read_bw[self.mle - 1]
# else:
#     step_cycles = max(write_bw_ll / self.mem_write_bw[self.mle - 1])
