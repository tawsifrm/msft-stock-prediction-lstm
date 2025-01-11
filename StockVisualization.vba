Sub VisualizeStockData()
    Dim ws As Worksheet
    Dim csvFile As String
    Dim lastRow As Long
    Dim chartObj As ChartObject
    Dim chartRange As Range

    ' Set the worksheet and CSV file path
    Set ws = ThisWorkbook.Sheets(1)
    csvFile = ThisWorkbook.Path & "\MicrosoftStock.csv"

    ' Clear existing data
    ws.Cells.Clear

    ' Import CSV data
    With ws.QueryTables.Add(Connection:="TEXT;" & csvFile, Destination:=ws.Range("A1"))
        .TextFileParseType = xlDelimited
        .TextFileConsecutiveDelimiter = False
        .TextFileTabDelimiter = False
        .TextFileSemicolonDelimiter = False
        .TextFileCommaDelimiter = True
        .TextFileSpaceDelimiter = False
        .TextFileColumnDataTypes = Array(1, 1, 1, 1, 1, 1, 1, 1)
        .Refresh BackgroundQuery:=False
    End With

    ' Find the last row of data
    lastRow = ws.Cells(ws.Rows.Count, "A").End(xlUp).Row

    ' Create a chart for actual stock prices
    Set chartObj = ws.ChartObjects.Add(Left:=100, Width:=375, Top:=50, Height:=225)
    Set chartRange = ws.Range("B2:B" & lastRow)
    With chartObj.Chart
        .SetSourceData Source:=chartRange
        .ChartType = xlLine
        .HasTitle = True
        .ChartTitle.Text = "Actual Stock Prices"
        .Axes(xlCategory, xlPrimary).HasTitle = True
        .Axes(xlCategory, xlPrimary).AxisTitle.Text = "Date"
        .Axes(xlValue, xlPrimary).HasTitle = True
        .Axes(xlValue, xlPrimary).AxisTitle.Text = "Price"
    End With

    ' Create a chart for predicted stock prices
    Set chartObj = ws.ChartObjects.Add(Left:=500, Width:=375, Top:=50, Height:=225)
    Set chartRange = ws.Range("C2:C" & lastRow)
    With chartObj.Chart
        .SetSourceData Source:=chartRange
        .ChartType = xlLine
        .HasTitle = True
        .ChartTitle.Text = "Predicted Stock Prices"
        .Axes(xlCategory, xlPrimary).HasTitle = True
        .Axes(xlCategory, xlPrimary).AxisTitle.Text = "Date"
        .Axes(xlValue, xlPrimary).HasTitle = True
        .Axes(xlValue, xlPrimary).AxisTitle.Text = "Price"
    End With
End Sub
