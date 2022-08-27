import ftplib
import shutil

"""
diff file format
aa.c:main\CMN\LP\FAULt\C
bb.c:main\CMN\LP\FAULt\C
"""

class SV_file_manager:

    m_var = {}

# ---------------------------------------------------------------------------
    def init_conf_dict(self):

        self.m_var['REMOTE_DEST_PATH'] = None  # remote base path
        self.m_var['FTP_USER'] = None          # FTP server username
        self.m_var['PJ_BASE_PATH'] = None      # local project base path
        self.m_var['FTP_PASS'] = None          # FTP server password
        self.m_var['FTP_SERVER'] = None        # FTP server IP
        self.m_var['LOCAL_DEST_PATH'] = None   # local path for diff file
        self.m_var['DEL_OBJ'] = 0              # del same name obj when upload .c file
        self.m_var['OBJ_BATH_PATH'] = None     # remote obj base path
        self.m_var['DIFF_LIST'] = None         # diff file list full path
        self.m_var['COPY_FM_PJ'] = None        # copy source files from PJ_BASE_PATH's project to LOCAL_DEST_PATH
        self.m_var['INVERSE_COPY'] = None      # copy source files from LOCAL_DEST_PATH to PJ_BASE_PATH's project
        self.m_var['NOT_UPLOAD'] = 0           # only copy the fils,do not upload
        self.m_var['CWF_DEBUG'] = None         # for debug
        self.m_var['BE_QUITE'] = None           # skip input

        return

# ---------------------------------------------------------------------------
    def read_conf_file(self, v_file_name):

        self.init_conf_dict()

        v_file = open(v_file_name, 'r')
        v_lines = v_file.readlines()

        for v_line in v_lines:
            v_line = v_line.splitlines()
            list_data = v_line[0].split('=', 2)
            if len(list_data) >= 2 and list_data[0] != '' and list_data[1] != '':
                self.m_var[list_data[0]] = list_data[1]
            else:
                print(str.format('config file line {} error\r\n', v_line))

        if self.m_var['CWF_DEBUG'] == '1':
            print(self.m_var)

        if self.m_var['NOT_UPLOAD'] == '1':
            print('Do not upload!!\r\n')

        if self.m_var['INVERSE_COPY'] == '1':
            print('!!!! This is a INVERSE_COPY(Local Dest -> PJ Path) !!!!!\r\n')
        else:
            print('!!!! COPY Direction(PJ Path -> Local Dest) !!!!!\r\n')

        if self.m_var['BE_QUITE'] == '1':
            # skip input
            return

        c_in = input(str.format('PJ path:{} is right? q to exit(do nothing)\r\n', self.m_var['PJ_BASE_PATH']))
        if c_in == 'q':
            exit(0)

        c_in = input(str.format('Local dest path:{} is right? q to exit(do nothing)\r\n', self.m_var['LOCAL_DEST_PATH']))
        if c_in == 'q':
            exit(0)

        c_in = input(str.format('Remote path:{} is right? q to exit(do nothing)\r\n', self.m_var['REMOTE_DEST_PATH']))
        if c_in == 'q':
            exit(0)

        c_in = input(str.format('FTP Server:{} is right? q to exit(do nothing)\r\n', self.m_var['FTP_SERVER']))
        if c_in == 'q':
            exit(0)

        c_in = input(str.format('Diff list:{} is right? q to exit(do nothing)\r\n', self.m_var['DIFF_LIST']))
        if c_in == 'q':
            exit(0)
        
        return

# ---------------------------------------------------------------------------
    def read_diff_list(self, v_file_name):

        all_data = []
        try:
            v_file = open(v_file_name, 'r')
        except:
            print('read_diff_list:File read fail!\r\n')
            raise FileExistsError

        print('read_diff_list:File read success!\r\n')

        # read lines in the file.
        v_lines = v_file.readlines()

        for v_line in v_lines:
            v_line = v_line.splitlines()
            list_data = v_line[0].split(':')
            all_data.append(list_data)

        print(all_data)
        return all_data

# ---------------------------------------------------------------------------
    def initial_ftp(self, v_ftp_server, v_user, v_pass):

        try:
            v_ftp_host = ftplib.FTP(v_ftp_server)
            v_ftp_host.login(v_user, v_pass)
        except:
            v_ftp_host = None
            raise

        return v_ftp_host

# ---------------------------------------------------------------------------
    def upload_ftp(self, v_file_path, v_file_name, v_dest_path, v_ftp_host):

        if v_file_name[-1] == '\\':
            v_full_name = str.format('{}{}', v_file_path, v_file_name)
        else:
            v_full_name = str.format('{}\\{}', v_file_path, v_file_name)

        try:
            v_ftp_host.cwd(v_dest_path)
        except:
            print(str.format('FTP CWD Fail(Path:{} , File:{})!\r\n', v_file_path, v_file_name))
            raise

        try:
            localfile = open(v_full_name, 'rb')
        except:
            print(str.format('Local file open Fail(Path:{} , File:{})!\r\n', v_file_path, v_file_name))
            raise

        ftp_cmd = str.format('STOR {}', v_file_name)

        try:
            v_ftp_host.storbinary(ftp_cmd, localfile)
        except:
            print(str.format('FTP Upload Fail(Path:{} , File:{})!\r\n', v_file_path, v_file_name))
            localfile.close()
            raise

        localfile.close()
        print(str.format('{} upload to {} success!\r\n', v_full_name, v_dest_path))
        return

# ---------------------------------------------------------------------------
    def do_job_upload(self):

        v_succ_index = 0
        v_fail_index = 0

        # Read Diff List File and get diff file list
        try:
            diff_data = self.read_diff_list(self.m_var['DIFF_LIST'])
        except FileExistsError:
            print('Diff file read fail!\r\n')
            raise
        except:
            print('Unknown Err!\r\n')
            raise
        else:
            print('Diff file read Success!\r\n')

        if self.m_var['COPY_FM_PJ'] == '1':
            # copy files from local PJ
            self.get_files_from_local_project(diff_data)

        if self.m_var['NOT_UPLOAD'] == '1':
            # do not upload files to remote
            return

        # initial FTP instance
        try:
            ftp_ser = self.initial_ftp(self.m_var['FTP_SERVER'], self.m_var['FTP_USER'], self.m_var['FTP_PASS'])
        except:
            print('FTP link fail\r\n')
            raise

        # upload files
        for v_file in diff_data:

            if self.m_var['REMOTE_DEST_PATH'][-1] != '/' and v_file[1][0] != '/':
                v_dest_full_path = str.format('{}/{}', self.m_var['REMOTE_DEST_PATH'], v_file[1])
            else:
                v_dest_full_path = str.format('{}{}', self.m_var['REMOTE_DEST_PATH'], v_file[1])

            try:
                self.upload_ftp(self.m_var['LOCAL_DEST_PATH'], v_file[0], v_dest_full_path, ftp_ser)
                v_succ_index = v_succ_index + 1
            except:
                print(str.format('{} to {} upload fail!\r\n', v_file[0], v_dest_full_path))
                v_fail_index = v_fail_index + 1
                continue

        # close FTP
        ftp_ser.close()

        # print result
        print(str.format('{} file upload success!\r\n', v_succ_index))
        print(str.format('{} file upload fail!\r\n', v_fail_index))

        return
    
# ---------------------------------------------------------------------------
    def get_files_from_local_project(self, diff_data):

        copy_success = 0
        copy_fail = 0

        if (self.m_var['PJ_BASE_PATH'] is None) or (self.m_var['PJ_BASE_PATH'] == ''):
            pj_base_path = ''
        elif self.m_var['PJ_BASE_PATH'][-1] != '\\':
            pj_base_path = str.format('{}\\', self.m_var['PJ_BASE_PATH'])
        else:
            pj_base_path = self.m_var['PJ_BASE_PATH']

        if self.m_var['LOCAL_DEST_PATH'][-1] != '\\':
            pj_local_dest = str.format('{}\\', self.m_var['LOCAL_DEST_PATH'])
        else:
            pj_local_dest = self.m_var['LOCAL_DEST_PATH']

        for v_file in diff_data:
            # replace '/' with '\'
            pj_file_path = v_file[1]
            pj_file_path = str.replace(pj_file_path, '/', '\\')

            old_name = str.format('{}{}{}', pj_base_path, pj_file_path, v_file[0])
            new_name = str.format('{}{}', pj_local_dest, v_file[0])

            try:
                if self.m_var['INVERSE_COPY'] == '1':
                    shutil.copyfile(new_name, old_name)
                    print(str.format('copy from {} to {} success!\r\n', new_name, old_name))
                else:
                    shutil.copyfile(old_name, new_name)
                    print(str.format('copy from {} to {} success!\r\n', old_name, new_name))
      
                copy_success = copy_success + 1
            except:
                if self.m_var['INVERSE_COPY'] == '1':
                    print(str.format('copy from {} to {} fail!\r\n', new_name, old_name))
                else:
                    print(str.format('copy from {} to {} fail!\r\n', old_name, new_name))

                copy_fail = copy_fail + 1
                raise

        # print result
        print(str.format('{} file copy success\r\n', copy_success))
        print(str.format('{} file copy fail\r\n', copy_fail))
        return
# ---------------------------------------------------------------------------
# class SV_file_manager over





# noinspection PyBroadException
c_str = ''
c_input = None
test_ins = SV_file_manager()

try:
    test_ins.read_conf_file('D:\\01_SV_file_manager\\conf_file.txt')
except:
    print('Conf file read fail!\r\n')
else:
    print('Conf file read success!\r\n')


if test_ins.m_var['INVERSE_COPY'] == '1':
    
    c_str = str.format('Cpoy File From {} to {}. Type \'q\' to quit(It will do nothing).', test_ins.m_var['LOCAL_DEST_PATH'], test_ins.m_var['REMOTE_DEST_PATH'])
    
elif test_ins.m_var['COPY_FM_PJ'] == '1':
    
    c_str = str.format('Cpoy File From {} to {}. Type \'q\' to quit(It will do nothing).', test_ins.m_var['PJ_BASE_PATH'], test_ins.m_var['REMOTE_DEST_PATH'])
    
else:
    
    c_str = str.format('Cpoy File From {} to {}. Type \'q\' to quit(It will do nothing).', test_ins.m_var['LOCAL_DEST_PATH'], test_ins.m_var['REMOTE_DEST_PATH'])

c_input = input(c_str)

if c_input == 'q':
    exit(0)
else:
    pass

test_ins.do_job_upload()

# to watch result info,waiting
input('Exit?')



