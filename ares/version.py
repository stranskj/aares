import time, datetime


def version_by_pkg():
    try:
        import pkg_resources
        return pkg_resources.get_distribution('ares')

    except ImportError:
        return 'Unknown, something is wrong with the Python installation'


try:
    import setuptools_scm #, git, os
    got_version = setuptools_scm.get_version(root='..',relative_to=__file__)

except ImportError:
    got_version = version_by_pkg()

except Exception as e:
    got_version = version_by_pkg() 
    #'\nDevelopment version. Cannot get proper state,\n please install Git, and setuptools_scm and gitpython package.'

  #  logging.exception(e) TODO: incoment when logging in place
    
version = got_version

if __name__ == "__main__":
    print('Version: {}'.format(version))
