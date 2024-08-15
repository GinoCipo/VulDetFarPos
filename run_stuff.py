import subprocess

# List of vulnerabilities
vulnerabilities = ["command_injection", "open_redirect", "path_disclosure", "remote_code", "sql", "xsrf", "xss"]

# Iterate over the vulnerabilities and run the command for each
for vulnerability in vulnerabilities:
    command = f"python Train-model.py {vulnerability} > log_{vulnerability}.txt"
    try:
        # Execute the command
        subprocess.run(command, check=True, shell=True)
        print("\n")
        print("---" * 10)
        print(f"Successfully ran command: {command}")
        print("---" * 10)
        print("\n")
    except subprocess.CalledProcessError as e:
        print("\n")
        print("---" * 10)
        print(f"Error running command: {command}")
        print(e)
        print("---" * 10)
        print("\n")
