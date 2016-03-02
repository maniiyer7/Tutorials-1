#!/usr/bin/python
###############################################################################
__author__ = 'amirkavousian'
__email__ = 'amir.kavousian@sunrun.com'
# created on: October 06, 2015
# summary: exploring Google API
###############################################################################

### STANDARD MODULES
import gdata
import gdata.service
import gdata.spreadsheet
import gdata.spreadsheet.service
import gdata.docs
import gdata.docs.service

# OTHER MODULES


# Create a client class which will make HTTP requests with Google Docs server.
client = gdata.docs.service.DocsService()
client.source
# Authenticate using your Google Docs email address and password.
# NOTE: client login is deprecated: https://developers.google.com/gdata/docs/auth/clientlogin
# client.ClientLogin('kav.amir@gmail.com', 'ebnofyamin34')  # , client.source

# Query the server for an Atom feed containing a list of your documents.
documents_feed = client.GetDocumentListFeed()

# Loop through the feed and extract each document entry.
for document_entry in documents_feed.entry:
  # Display the title of the document on the command line.
  print document_entry.title.text



###############################################################################
###############################################################################
###############################################################################
### From" monthly metrics file

# import pApi, pOrcl, pMysql
import os, pytz
from datetime_tut import datetime_tut as dt
import datetime_tut as dat
import pCommon as com

import gdata.spreadsheet.text_db
import gdata.gauth


unixFormat='%Y-%m-%d %H:%M:%S'
oracleFormat='%Y%m%d'
googleFormat='%m/%d/%Y'
oracleFormatMonth='%Y%m'
timestampTimezone = 'America/Los_Angeles'
timestampFormat = '%Y-%m-%d %H:%M:%S %Z'

email = 'amir.kavousian@sunrunhome.com'
password = 'Sohr3vard3'
spreadsheet_name ='Test Spreadsheet'


client = gdata.spreadsheet.text_db.DatabaseClient(username=email, password=password)
database = client.GetDatabases(name=spreadsheet_name)


from gdata.spreadsheet.service import SpreadsheetsService
key = '0Aip8Kl9b7wdidFBzRGpEZkhoUlVPaEg2X0F2YWtwYkE'
client = SpreadsheetsService()
feed = client.GetWorksheetsFeed(key, visibility='public', projection='basic')

for sheet in feed.entry:
  print sheet.title.text


#######################################
def update_Table_Value(database, worksheetName, field, date, val):
    table = database.GetTables(name=worksheetName)
    tb = table[0].GetRecords(1,200)
    if field not in tb[0].content.keys():
        print 'field: '+field+' is not in table'
        print 'valid keys are: '+str(tb[0].content.keys())
        return False
    for e in tb:
        if e.content['month']==date:
            e.content[field] = val
            e.Push()
            return True
    print 'row: '+date+' is not in table'
    return True
#######################################


#######################################
def google_Format(ts):
    return str(ts.month)+'/'+str(ts.day)+'/'+str(ts.year)
#######################################


#######################################
# GET METRICS

validAssetStatus = list(pMysql.get_valid_asset_status_states()['asset_status'])

today = dat.date.today()
first = dat.date(day=1, month=today.month, year=today.year)
ldlm = first - dat.timedelta(days=1)
fdlm = dat.date(day=1, month=ldlm.month, year=ldlm.year)


alertTypes = [{'alert_name':'Non Comms','noc_int_ref':['Non Comm'],'oracle_case_reason':'Metering'},
              {'alert_name':'System Failure','noc_int_ref':['System Failure'],'oracle_case_reason':'System Failure'},
              {'alert_name':'System Underperformance','noc_int_ref':['System Underperformance','Startup Underperformance'],'oracle_case_reason':'System Under-performance'},]
#######################################


#######################################
def get_Alert_Balance_Metrics():
    # return the percent of alerts for the month that were turned into cases for each alert type

    arr = []
    for alertType in alertTypes:
        sql = """SELECT DISTINCT(service_contract_name)
                FROM noc_alerts a, noc_alert_type b
                WHERE created<='"""+ldlm.strftime(unixFormat)+"""' and created>='"""+fdlm.strftime(unixFormat)+"""'
                    AND service_contract_name IS NOT NULL
                    AND a.alert_type_id=b.alert_type_id
                    AND b.int_ref IN ('"""+"','".join(alertType['noc_int_ref'])+"""')
                """
        alertServiceContracts = list(pMysql.user_defined_query(sql,[('service_contract_name','O')])['service_contract_name'])
        chunks = com.chunk_list(alertServiceContracts,1000)

        sql = """SELECT DISTINCT(a.service_contract_name)
                FROM dim_asset a, dim_case b
                WHERE b.case_service_contract_id=a.service_contract_id
                    AND b.case_record_type='FS Ticket'
                    AND b.case_open_date>TO_TIMESTAMP('"""+fdlm.strftime(oracleFormat)+"""', 'YYYYMMDD')
                    AND b.case_open_date<=TO_TIMESTAMP('"""+ldlm.strftime(oracleFormat)+"""', 'YYYYMMDD')
                    AND (a.service_contract_name IN ("""+""") OR a.service_contract_name IN (""".join(["'"+"','".join(chunk)+"'" for chunk in chunks])+"""))
                    AND b.case_reason='"""+alertType['oracle_case_reason']+"""'
                """

        caseServiceContracts = list(pOrcl.user_defined_query(sql, [('service_contract_name','O')])['service_contract_name'])
        alertsCased = sum([1 for x in alertServiceContracts if x in caseServiceContracts])
        arr.append({'alert_name':alertType['alert_name'],'alert_balance':alertsCased/float(len(alertServiceContracts)),'sys_alerts_created':len(alertServiceContracts),'sys_alert_cases_created':alertsCased}  )

    return arr

def get_Monthly_Production(withContract=False):
    # get expected and actual production for the last month

    sql = """SELECT /*+ PARALLEL(8) */ SUM(d.generation), SUM(e.cust_contract_scaled_gen)
            FROM dim_asset a, fact_asset_management b, dim_product_type c, fact_repgen_daily d, fact_expected_gen_daily e
            WHERE d.generation_date_fkey=e.generation_date_fkey
            AND d.d_asset_fkey=e.d_asset_fkey
            AND a.d_asset_key=d.d_asset_fkey
            AND a.d_asset_key=b.d_asset_fkey
            AND b.d_product_type_fkey=c.d_product_type_key
            AND c.agreement_type NOT IN ('"""+"','".join(notMonitoringAgreements)+"""')
            """+("AND e.cust_contract_scaled_gen!=0" if withContract else "")+"""
            AND TO_CHAR(TO_TIMESTAMP(CAST(d.generation_date_fkey AS VARCHAR(8)), 'YYYYMMDD'),'YYYYMM')='"""+ldlm.strftime(oracleFormatMonth)+"'"

    resp = pOrcl.user_defined_query(sql, [('actual','O'),('cust_contract','O')])
    return {'actual':resp['actual'][0],'cust_contract':resp['cust_contract'][0]}

def get_Cumulative_Production(withContract=False):
    # get expected and actual production for all of time up till latest month

    sql = """SELECT /*+ PARALLEL(8) */ SUM(d.generation), SUM(e.cust_contract_scaled_gen)
            FROM dim_asset a, fact_asset_management b, dim_product_type c, fact_repgen_daily d, fact_expected_gen_daily e
            WHERE d.generation_date_fkey=e.generation_date_fkey
            AND d.d_asset_fkey=e.d_asset_fkey
            AND a.d_asset_key=d.d_asset_fkey
            AND a.d_asset_key=b.d_asset_fkey
            AND b.d_product_type_fkey=c.d_product_type_key
            AND c.agreement_type NOT IN ('"""+"','".join(notMonitoringAgreements)+"""')
            """+("AND e.cust_contract_scaled_gen!=0" if withContract else "")+"""
            AND TO_CHAR(TO_TIMESTAMP(CAST(d.generation_date_fkey AS VARCHAR(8)), 'YYYYMMDD'),'YYYYMM')<='"""+ldlm.strftime(oracleFormatMonth)+"'"

    resp = pOrcl.user_defined_query(sql, [('actual','O'),('cust_contract','O')])
    return {'actual':resp['actual'][0],'cust_contract':resp['cust_contract'][0]}

def get_Actionable_Alert_Metrics():
    # get number of dispatches made off alerts/cases created

    arr = []
    for alertType in alertTypes:
        sql = """SELECT
                    ROUND(SUM(CASE WHEN case_closed_date IS NOT NULL AND fs_dispatch_name IS NULL THEN 0 ELSE 1 END)/COUNT(*),3) acceptance_ratio,
                    COUNT(*) cases_identified,
                    SUM(case when fs_dispatch_name IS NOT NULL THEN 1 ELSE 0 END) dispatch_count
                FROM (
                    SELECT
                        a.case_number case_number,
                        a.case_open_date case_open_date,
                        a.case_closed_date case_closed_date,
                        a.case_status case_status,
                        a.case_reason case_reason,
                        a.case_origin case_origin,
                        a.case_record_type case_record_type,
                        a.case_service_contract_id case_service_contract_id,
                        b.fs_dispatch_name fs_dispatch_name
                    FROM
                        dim_case a
                    LEFT JOIN (
                        SELECT
                            *
                        FROM
                            fact_field_service_ticket b,
                            dim_fs_dispatch c where b.d_fs_dispatch_fkey=c.d_fs_dispatch_key
                        ) b ON a.d_case_key=b.d_case_fkey
                    WHERE a.case_record_type='FS Ticket'
                    AND TO_CHAR(a.case_open_date,'YYYYMM')='"""+ldlm.strftime(oracleFormatMonth)+"""'
                    AND a.case_reason='"""+alertType['oracle_case_reason']+"""'
                    AND a.case_origin='SunRun Staff'
                    ORDER BY a.case_open_date
                    ) a"""
        resp = pOrcl.user_defined_query(sql, [('acceptance_ratio','O'),('cases_identified','O'),('dispatch_count','O')])
        arr.append({'alert_name':alertType['alert_name'],'acceptance_ratio':resp['acceptance_ratio'][0]*100,'cases_identified':resp['cases_identified'][0],'dispatch_count':resp['dispatch_count'][0]}  )
    return arr

def get_Chronic_UPS_Metrics():

    threshold = 0.8
    arr = {}

    # cumulative metric
    sql = """SELECT /*+ PARALLEL(8) */ COUNT(*) cnt, ROUND(SUM(flag)/COUNT(*)*100,2) pct_chronic
            FROM(
                SELECT a.service_contract_name, d.flag
                FROM
                  dim_asset a,
                  fact_asset_management b,
                  dim_proposal c,
                  dim_product_type e,
                  (SELECT
                    e.d_asset_fkey,
                    CASE WHEN SUM(d.generation)/SUM(e.cust_contract_scaled_gen)<"""+str(threshold)+""" THEN 1 ELSE 0 END flag,
                    SUM(d.generation) generation,
                    SUM(e.cust_contract_scaled_gen) exp_generation
                  FROM
                    fact_repgen_daily d,
                    fact_expected_gen_daily e
                  WHERE
                    d.d_asset_fkey=e.d_asset_fkey
                    AND d.generation_date_fkey=e.generation_date_fkey
                    AND e.generation_date_fkey>"""+dat.datetime_tut(ldlm.year-1,ldlm.month,ldlm.day).strftime(oracleFormat)+"""
                    AND e.generation_date_fkey<="""+ldlm.strftime(oracleFormat)+"""
                    AND e.cust_contract_scaled_gen!=0
                  GROUP BY e.d_asset_fkey) d
                WHERE a.d_asset_key=b.d_asset_fkey
                  AND b.d_proposal_fkey=c.d_proposal_key
                  AND d.d_asset_fkey=a.d_asset_key
                  --AND ABS(a.estimated_year_1_generation-c.first_yr_gen_estimate_kwh)<=2 -- comment out this line to get project count and still ptoed, don't care about certainty
                  AND b.d_product_type_fkey=e.d_product_type_key
                  AND e.agreement_type NOT IN ('"""+"','".join(notMonitoringAgreements)+"""')
                  AND b.pto_date_fkey<="""+dat.datetime_tut(ldlm.year-1,ldlm.month,ldlm.day).strftime(oracleFormat)+"""
                  AND (a.asset_status in ('"""+"','".join(validAssetStatus)+"""') AND b.pto_date_fkey!=0)
                ORDER BY service_contract_name
            ) a"""
    resp = pOrcl.user_defined_query(sql, [('count','O'),('pct_chronic','O')])
    arr['cumulative'] = {'count':resp['count'][0],'pct':resp['pct_chronic'][0]}

    # consecutive metric
    sql = """SELECT /*+ PARALLEL(8) */ COUNT(*) cnt, ROUND(SUM(flag)/COUNT(*)*100,2) pct_chronic
            FROM(
                SELECT a.service_contract_name, d.flag
                FROM
                  dim_asset a,
                  fact_asset_management b,
                  dim_proposal c,
                  dim_product_type e,
                  (SELECT
                    a.d_asset_fkey,
                    CASE WHEN MAX(pi_ratio)<"""+str(threshold)+""" THEN 1 ELSE 0 END flag
                  FROM
                    (SELECT
                      e.d_asset_fkey,
                      TO_CHAR(TO_TIMESTAMP(CAST(e.generation_date_fkey AS VARCHAR(8)),'YYMMDD' ),'YYYY-MM') mth,
                      SUM(d.generation)/SUM(e.cust_contract_scaled_gen) pi_ratio,
                      SUM(d.generation) generation,
                      SUM(e.cust_contract_scaled_gen) exp_generation
                    FROM
                      fact_repgen_daily d,
                      fact_expected_gen_daily e
                    WHERE
                      d.d_asset_fkey=e.d_asset_fkey
                      AND d.generation_date_fkey=e.generation_date_fkey
                      AND e.generation_date_fkey>"""+dat.datetime_tut(ldlm.year-1,ldlm.month,ldlm.day).strftime(oracleFormat)+"""
                      AND e.generation_date_fkey<="""+ldlm.strftime(oracleFormat)+"""
                      AND e.cust_contract_scaled_gen!=0
                    GROUP BY e.d_asset_fkey, TO_CHAR(TO_TIMESTAMP(CAST(e.generation_date_fkey AS VARCHAR(8)),'YYMMDD' ),'YYYY-MM')
                    ORDER BY d_asset_fkey, mth) a
                  GROUP BY a.d_asset_fkey) d
                WHERE a.d_asset_key=b.d_asset_fkey
                  AND b.d_proposal_fkey=c.d_proposal_key
                  AND d.d_asset_fkey=a.d_asset_key
                  --AND ABS(a.estimated_year_1_generation-c.first_yr_gen_estimate_kwh)<=2 -- comment out this line to get project count and still ptoed, don't care about certainty
                  AND b.d_product_type_fkey=e.d_product_type_key
                  AND e.agreement_type NOT IN ('"""+"','".join(notMonitoringAgreements)+"""')
                  AND b.pto_date_fkey<="""+dat.datetime_tut(ldlm.year-1,ldlm.month,ldlm.day).strftime(oracleFormat)+"""
                  AND (a.asset_status in ('"""+"','".join(validAssetStatus)+"""') and b.pto_date_fkey!=0)
                ORDER BY service_contract_name
            ) a"""

    resp = pOrcl.user_defined_query(sql, [('count','O'),('pct_chronic','O')])
    arr['consecutive'] = {'count':resp['count'][0],'pct':resp['pct_chronic'][0]}

    # get project count of projects that we have confidence in their contracted FYGE (first year generation estimate)
    resp = pOrcl.user_defined_query(sql.replace('--AND ABS(a.estimated_year_1_generation', 'AND ABS(a.estimated_year_1_generation'), [('count','O'),('pct_chronic','O')])
    arr['consecutive']['corrected_count'] = resp['count'][0]
    arr['cumulative']['corrected_count'] = resp['count'][0]

    return arr

def get_Corporate_Metric():


    sql = """
            SELECT /*+ PARALLEL(8) */ ROUND(SUM(ups)/COUNT(*),4) ups_rate, COUNT(*) total, SUM(ups) ups
            FROM (
            SELECT
              a.service_contract_name service_contract_name,
              CASE WHEN d.actual/d.cust_contract<0.8 THEN 1 ELSE 0 END ups
            FROM
              dim_asset a,
              fact_asset_management b,
              dim_proposal c,
              (SELECT a.d_asset_key, SUM(d.generation) actual, SUM(e.cust_contract_scaled_gen) cust_contract
              FROM dim_asset a, fact_repgen_daily d, fact_expected_gen_daily e
              WHERE d.d_asset_fkey=a.d_asset_key
              AND e.d_asset_fkey=a.d_asset_key
              AND e.generation_date_fkey=d.generation_date_fkey
              AND d.generation_date_fkey>"""+dat.datetime_tut(ldlm.year-1,ldlm.month,ldlm.day).strftime(oracleFormat)+"""
              AND d.generation_date_fkey<="""+ldlm.strftime(oracleFormat)+"""
              GROUP BY a.d_asset_key) d,
              dim_facility_asset e,
              fact_summary_solar_facility f,
              dim_manufacturer g,
              dim_product_type h
            WHERE
              a.d_asset_key=b.d_asset_fkey
              AND d.d_asset_key=a.d_asset_key
              AND b.d_proposal_fkey=c.d_proposal_key
              AND b.d_facility_asset_fkey=e.d_facility_asset_key
              AND f.d_asset_fkey=a.d_asset_key
              AND g.d_manufacturer_key=f.d_inverter_manufacturer_fkey
              AND d.cust_contract!=0
              AND b.pto_date_fkey<="""+ldlm.strftime(oracleFormat)+"""
              AND ABS(a.estimated_year_1_generation-c.first_yr_gen_estimate_kwh)<=2
              AND (a.asset_status IN ('"""+"','".join(validAssetStatus)+"""') AND b.pto_date_fkey!=0)
              AND b.d_product_type_fkey=h.d_product_type_key
              AND h.agreement_type NOT IN ('"""+"','".join(notMonitoringAgreements)+"""')
            ) a
            """

    resp = pOrcl.user_defined_query(sql, [('ups_rate','O'),('total','O'),('ups','O')])
    arr ={'ups_rate':resp['ups_rate'][0]*100,'total':resp['total'][0],'ups':resp['ups'][0]}
    return arr

print dt.now(pytz.timezone(timestampTimezone)).strftime(timestampFormat)+': getting metrics data...'
print dt.now(pytz.timezone(timestampTimezone)).strftime(timestampFormat)+': getting alert balanced metric(s)...'
alertBalance = get_Alert_Balance_Metrics()
print dt.now(pytz.timezone(timestampTimezone)).strftime(timestampFormat)+': getting cumulative production metric(s)...'
cumulativeProduction = get_Cumulative_Production()
cumulativeProductionwContract = get_Cumulative_Production(True)
print dt.now(pytz.timezone(timestampTimezone)).strftime(timestampFormat)+': getting monthly production metric(s)...'
monthlyProduction = get_Monthly_Production()
monthlyProductionwContract = get_Monthly_Production(True)
print dt.now(pytz.timezone(timestampTimezone)).strftime(timestampFormat)+': getting actionable alerts metric(s)...'
actionableAlerts = get_Actionable_Alert_Metrics()
print dt.now(pytz.timezone(timestampTimezone)).strftime(timestampFormat)+': getting chronic ups metric(s)...'
chronicUPS = get_Chronic_UPS_Metrics()
print dt.now(pytz.timezone(timestampTimezone)).strftime(timestampFormat)+': getting corporate metric(s)...'
corpMetrics = get_Corporate_Metric()


# COPY METRICS TO GOOGLE
print dt.now(pytz.timezone(timestampTimezone)).strftime(timestampFormat)+': entering metrics...'
######



worksheet_name = 'Fleet Performance inputs'
# performance metrics with no contracts
update_Table_Value(database[0], worksheet_name, 'fleetperformancemonthlyactual', google_Format(fdlm), str(monthlyProduction['actual']))
update_Table_Value(database[0], worksheet_name, 'fleetperformancemonthlyexpectedas-built', google_Format(fdlm), str(monthlyProduction['cust_contract']))
update_Table_Value(database[0], worksheet_name, 'fleetperformancecumulativeactual', google_Format(fdlm), str(cumulativeProduction['actual']))
update_Table_Value(database[0], worksheet_name, 'fleetperformancecumulativeexpectedas-built', google_Format(fdlm), str(cumulativeProduction['cust_contract']))

# performance metrics with valid contracts
update_Table_Value(database[0], worksheet_name, 'fleetperformancemonthlyactualwvalidcontract', google_Format(fdlm), str(monthlyProductionwContract['actual']))
update_Table_Value(database[0], worksheet_name, 'fleetperformancemonthlyexpectedvalidcontractas-built', google_Format(fdlm), str(monthlyProductionwContract['cust_contract']))
update_Table_Value(database[0], worksheet_name, 'fleetperformancecumulativeactualwvalidcontract', google_Format(fdlm), str(cumulativeProductionwContract['actual']))
update_Table_Value(database[0], worksheet_name, 'fleetperformancecumulativeexpectedvalidcontractas-built', google_Format(fdlm), str(cumulativeProductionwContract['cust_contract']))

update_Table_Value(database[0], worksheet_name, 'percentofsystemsunder-performingmorethan20actual', google_Format(fdlm), str(corpMetrics['ups_rate']))
update_Table_Value(database[0], worksheet_name, 'chronicunder-performanceconsecutiveactual', google_Format(fdlm), str(chronicUPS['consecutive']['pct']))
update_Table_Value(database[0], worksheet_name, 'chronicunder-performancecumulativeactual', google_Format(fdlm), str(chronicUPS['cumulative']['pct']))
update_Table_Value(database[0], worksheet_name, 'projectcountasofendofmonthyearagowconfidentcontractandstillptoed', google_Format(fdlm), str(chronicUPS['consecutive']['corrected_count']))
update_Table_Value(database[0], worksheet_name, 'projectcountasofendofmonthyearagoandstillptoed', google_Format(fdlm), str(chronicUPS['consecutive']['count']))

worksheet_name = 'Monitoring inputs'
for al in actionableAlerts:
    if al['alert_name']=='Non Comms':
        update_Table_Value(database[0], worksheet_name, 'actionablealertratemeteringactual', google_Format(fdlm), str(al['acceptance_ratio']))
    if al['alert_name']=='System Failure':
        update_Table_Value(database[0], worksheet_name, 'actionablealertratesystemfailureactual', google_Format(fdlm), str(al['acceptance_ratio']))
    if al['alert_name']=='System Underperformance':
        update_Table_Value(database[0], worksheet_name, 'actionablealertrateunder-performanceactual', google_Format(fdlm), str(al['acceptance_ratio']))

for al in alertBalance:
    if al['alert_name']=='Non Comms':
        update_Table_Value(database[0], worksheet_name, 'uniquesystemalertscreatednon-comm', google_Format(fdlm), str(al['sys_alerts_created']))
        update_Table_Value(database[0], worksheet_name, 'uniquesystemcasescreatedwalertsnon-comm', google_Format(fdlm), str(al['sys_alert_cases_created']))
    if al['alert_name']=='System Failure':
        update_Table_Value(database[0], worksheet_name, 'uniquesystemalertscreatedsystemfailure', google_Format(fdlm), str(al['sys_alerts_created']))
        update_Table_Value(database[0], worksheet_name, 'uniquesystemcasescreatedwalertssystemfailure', google_Format(fdlm), str(al['sys_alert_cases_created']))
    if al['alert_name']=='System Underperformance':
        update_Table_Value(database[0], worksheet_name, 'uniquesystemalertscreatedunderperforming', google_Format(fdlm), str(al['sys_alerts_created']))
        update_Table_Value(database[0], worksheet_name, 'uniquesystemcasescreatedwalertsunderperforming', google_Format(fdlm), str(al['sys_alert_cases_created']))


print 'done'
print '<---- finished script '+os.path.basename(__file__)+' '+dt.now(pytz.timezone(timestampTimezone)).strftime(timestampFormat)+' ---->'

