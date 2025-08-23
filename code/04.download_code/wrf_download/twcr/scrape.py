import scrapy

class MySpider(scrapy.Spider):
    name = "netcdf_spider"
    allowed_domains = ["downloads.psl.noaa.gov"]
    start_urls = ["https://downloads.psl.noaa.gov/Datasets/20thC_ReanV3/"]
    def parse(self, response):
       # folder to download for WRF
        # fol=["2m","10m","accum","misc","prs","sfc","subsfc"]
       # folder to download for compare with synop
        fol = ["2m","10m","accum"]
        # Find all the links on the page
        links = response.css('a::attr(href)').extract()

        # Loop through each link and check if it contains "SI/" and "MO/":
        for link in links:
            #if any(type in link for type in ["SI/","MO/"]):
            if link.endswith(("MO/","SI/")) and any(f in link for f in fol) :	            
   		 # Combine the link with the base URL to get the full URL
                full_url = response.urljoin(link)
                print(full_url)
                # Make a request to the URL and pass the response to the parse_small method
                yield scrapy.Request(full_url, callback=self.parse_small)

    def parse_small(self, response):
        # Find all the links on the page
          # Full variable to download
        #var = ["air", "rhum", "shum", "uwnd", "vwnd", "hgt", "pres", "prmsl",\
        #       "skt", "soilm", "tsoil", "apcp"]
          # Variable to compare with synop
        var = ["tmax","tmin","vwnd","uwnd","apcp"] 
        #year = [1877,1878,1887, 1889, 1930, 1945, 1954, 1971, 1972]
        links = response.css('a::attr(href)').extract()


        # Loop through each link and check if it ends with ".nc"
        for link in links:
            # if link.endswith('.nc') \
              #  and int(link.split('.')[-2]) in year \
              #  and any(v in link for v in var):
            if link.endswith('.nc') \
               and 1876 < int(link.split('.')[-2]) < 2015 \
               and any(v in link for v in var):
                print(link)
                # Combine the link with the base URL to get the full URL of the NetCDF file
                full_url = response.urljoin(link)
                # print(full_url)
                if ("tropo" in full_url and any (v in link for v in ["air","uwnd","vwnd","hgt"])) \
                   or ("hgtAbvSfc" in full_url) \
                   or ("misc" in full_url and ("rhum" in link)) \
                   or ( "prs" in full_url and "rhum" in link):
                     continue
                yield {
                    'file_url': full_url
                }
