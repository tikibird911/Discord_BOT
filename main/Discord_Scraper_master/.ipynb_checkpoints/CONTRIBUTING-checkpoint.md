## Contributing to this project:
If you wish to contribute to this project then you can simply create a
pull request and I'll look through the changes and test them out on my
system before merging the pull request.

## Code of Conduct:
By using this script you are agreeing to the following stipulalations:
* I will only use this script for archival purposes.
* Dracovian holds no liability or responsibility for any misuse of this script.

  If you don't intend to use this script and you happen to be a Discord staff member
then I strongly advise you to check out the Wiki for this project since I have layed
out my thoughts on the TOS regarding self-botting regarding how it makes the process
of archiving information nearly impossible as-is.

## Creating an issue:
There is currently four labels when creating an issue:
* Outdated
* Download Error
* Runtime Error
* Vulnerable

This is how these labels are intended for use:
### Outdated
This label is only appropriate if...
* Python 4 magically comes out and Python 2 and 3 get updated to remove native urllib2/http.client support.
* Discord releases v7 of their API rendering their current version obsolete.

Anything else that doesn't cause an issue with the runtime of the script can be labelled as vulnerable if
it meets the intended use of the vulnerable label.

### Download Error
This label is only appropriate if...
* There are files that are partially downloaded (corrupted images).
* There are errors regarding the creation of files/folders (permissions or file/folder naming).
* There are permission errors when downloading files (HTTP 4XX errors).

If the download error comes with some text in the commandline then this is likely a Runtime Error as well so
be sure to push that label and follow the intended use for the Runtime Error label.

### Runtime Error
This label is only appropriate if...
* There is an error message popping up in your commandline.

If there is no error message popping up then it's probably a Download Error or an Outdated error. Also be sure
to supply the contents of the error message in your issue (just make sure to remove any personable information that
may come with said error message).

### Vulnerable
This label is only appropriate if...
* A function or module creates a point of exploit on the host system.
* The script somehow gets picked up by an antivirus program as malicious/PUP.

So far I am not expecting this label to be used anytime soon, but in case there is such an error then I have made this
label as a way to quickly get my attention since the last thing I want is to write a script that can harm people's systems.
