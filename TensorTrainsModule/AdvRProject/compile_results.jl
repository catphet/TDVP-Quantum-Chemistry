# Redirect output to both terminal and file
output_file = open("compiled_results.txt", "w")  

function logprint(args...)
    
    println(output_file, args...)
    flush(output_file)
end


files = ["results_n3.txt", "results_n5.txt", "results_n10.txt", "results_tang_n3.txt", "results_tang_n5.txt", "results_tang_n10.txt"]

for f in files
    if isfile(f)
        logprint(output_file, "="^60)
        logprint(output_file, "FILE: $f")
        logprint(output_file, "="^60)
        logprint(output_file, read(f, String))
        logprint(output_file, "\n")
    else
        logprint(output_file, "FILE NOT FOUND: $f")
    end
end

close(output_file)
logprint("Done! See compiled_results.txt")