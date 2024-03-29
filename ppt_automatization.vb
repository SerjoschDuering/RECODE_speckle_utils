#If VBA7 Then
    Public Declare PtrSafe Sub Sleep Lib "kernel32" (ByVal dwMilliseconds As LongPtr) 'For 64 Bit Systems
#Else
    Public Declare Sub Sleep Lib "kernel32" (ByVal dwMilliseconds As Long) 'For 32 Bit Systems
#End If

Option Explicit


Function IndexOf(col As collection, val As Variant) As Integer
    Dim i As Integer
    For i = 1 To col.Count
        If col(i) = val Then
            IndexOf = i
            Exit Function
        End If
    Next i
    IndexOf = -1
End Function

Function ParseText(line As String) As collection
    Dim regex As Object
    Set regex = CreateObject("VBScript.RegExp")
    regex.Global = True
    regex.Pattern = "<([^>]+)>([^<]+)"

    Dim matches As Object
    Set matches = regex.Execute(line)

    Dim collection As New collection

    Dim match As Object
    For Each match In matches
        Dim dict As Object
        Set dict = CreateObject("Scripting.Dictionary")
        Dim style As String
        style = match.SubMatches(0)
        Dim styleAttributes As Variant
        styleAttributes = Split(style, ",")
        dict("font") = Trim(styleAttributes(0))
        dict("size") = CInt(Trim(styleAttributes(1)))
        dict("style") = Trim(styleAttributes(2))
        dict("text") = match.SubMatches(1)
        collection.Add dict
    Next match

    Set ParseText = collection
End Function

Function ParseTableText(line As String) As collection
    ' Split the line into individual cell information
    Dim cellInfos As Variant
    cellInfos = Split(line, "]")
    
    Dim collection As New collection

    Dim cellInfo As Variant
    For Each cellInfo In cellInfos
        ' Ignore empty strings
        If Trim(cellInfo) <> "" Then
            ' Remove the opening bracket
            cellInfo = Replace(cellInfo, "[", "")
            
            ' Split the cell info into location and text
            Dim locAndText As Variant
            locAndText = Split(cellInfo, " ", 2)
            
            ' Remove the parentheses
            Dim locString As String
            locString = Replace(locAndText(0), "(", "")
            locString = Replace(locString, ")", "")
            
            ' Split the location into row and column
            Dim loc As Variant
            loc = Split(locString, ",")
            
            ' Debug statements to inspect the variables
            Debug.Print "locAndText: "; locAndText(0)
            'Debug.Print "loc: "; loc(0)
            
            ' Create a dictionary and add it to the collection
            Dim dict As Object
            Set dict = CreateObject("Scripting.Dictionary")
            dict("row") = CInt(Trim(loc(1)))
            dict("col") = CInt(Trim(loc(0)))
            dict("text") = locAndText(1)
            collection.Add dict
        End If
    Next cellInfo

    Set ParseTableText = collection
End Function



Sub ApplyTextStyle(textRange As textRange, collection As collection)
    Dim dict As Object
    Dim startPos As Long
    startPos = 1
    
    ' Clear the existing text
    textRange.text = ""
    
    For Each dict In collection
        Dim text As String
        text = dict("text")
        
        ' Append the formatted text to the existing text range
        textRange.InsertAfter text
        With textRange.Characters(startPos, Len(text)).Font
            .Name = dict("font")
            .Size = dict("size")
            '.Allcaps = msoFalse ' Add this line to disable All Caps
            '.Smallcaps = msoFalse ' Add this line to disable Small Caps
            If dict("style") = "bold" Then
                .Bold = msoTrue
            Else
                .Bold = msoFalse
            End If
            If dict("style") = "italic" Then
                .Italic = msoTrue
            Else
                .Italic = msoFalse
            End If
        End With
        
        startPos = startPos + Len(text)
    Next dict
End Sub




Sub ReplaceObjects()

    Dim fso As New FileSystemObject
    Dim folder As folder
    Dim subfolder As folder
    Dim file As file
    Dim filePaths As New collection
    Dim fileNames As New collection
    Dim shapeZOrder As New Scripting.Dictionary
    
    ' log file
    Dim logFile As textStream
    Dim logFileName As String
    Const MAX_PATH As Integer = 260
    Dim imagesReplaced As Integer
    Dim textsReplaced As Integer
    Dim tablesReplaced As Integer
    Dim noFileMatched As Integer
    
    ' Initialize counters
    imagesReplaced = 0
    textsReplaced = 0
    tablesReplaced = 0
    noFileMatched = 0
    
    
    

    ' 1. Prompt for a directory
    With Application.FileDialog(msoFileDialogFolderPicker)
        .Title = "Select Parent Directory"
        .AllowMultiSelect = False
        If .Show = -1 Then ' If OK is pressed
            Set folder = fso.GetFolder(.SelectedItems(1))
            logFileName = folder.Path & "\LogFile.txt" ' Create log file in selected directory
            Set logFile = fso.CreateTextFile(logFileName, True)
            logFile.WriteLine "Files with too long filepaths:"
        Else
            Exit Sub
        End If
    End With
    
    
   

    ' 2. Index all files within the directory and its subdirectories
    Debug.Print "Indexing files..."
    For Each file In folder.Files
          If Len(file.Path) > MAX_PATH Then
                logFile.WriteLine file.Path
            Else
                filePaths.Add file.Path
                fileNames.Add file.Name
            End If
    Next file
    For Each subfolder In folder.SubFolders
        For Each file In subfolder.Files
            If Len(file.Path) > MAX_PATH Then
                logFile.WriteLine file.Path
            Else
                filePaths.Add file.Path
                fileNames.Add file.Name
            End If
        Next file
    Next subfolder
    logFile.WriteLine "-"
    logFile.WriteLine "unmatched files:"
    ' 3. Iterate through all slides and objects
    Dim idx As Integer
    'Debug.Print "Iterating slides..."
    Dim slide As PowerPoint.slide
    For Each slide In ActivePresentation.Slides
        ' Clear the ZOrder dictionary for each new slide
        shapeZOrder.RemoveAll

        'Debug.Print "Iterating shapes in slide " & slide.SlideNumber
        Dim shape As PowerPoint.shape
        For Each shape In slide.Shapes
            ' Save the ZOrder for all shapes (not just those to be replaced)
            Dim shapeKey As String
            shapeKey = shape.Name & "-" & shape.ZOrderPosition
            shapeZOrder.Add shapeKey, shape.ZOrderPosition
        Next shape

        ' Iterate again for replacement
        For Each shape In slide.Shapes
            If Left(shape.Name, 2) = "#+" Then ' if the object is to be replaced
                'Debug.Print "Checking shape " & shape.Name
                idx = IndexOf(fileNames, Mid(shape.Name, 3)) ' get the index of the file by name
                If idx <> -1 Then ' if file is found
                    'Debug.Print "a matching file was found "
                    ' check if it's a picture or a text file
                    If Right(fileNames.Item(idx), 3) = "png" Or Right(fileNames.Item(idx), 3) = "jpg" Then
                        If shape.Type = msoPicture Or shape.Type = msoLinkedPicture Then ' if shape is a picture
                            ' keep old picture properties
                            Dim oldCropTop As Single: oldCropTop = shape.PictureFormat.CropTop
                            Dim oldCropBottom As Single: oldCropBottom = shape.PictureFormat.CropBottom
                            Dim oldCropLeft As Single: oldCropLeft = shape.PictureFormat.CropLeft
                            Dim oldCropRight As Single: oldCropRight = shape.PictureFormat.CropRight

                            ' remove cropping before deleting
                            With shape.PictureFormat
                                .CropTop = 0
                                .CropBottom = 0
                                .CropLeft = 0
                                .CropRight = 0
                            End With

                            ' image size without cropping
                            Dim oldName As String: oldName = shape.Name
                            Dim oldLeft As Single: oldLeft = shape.Left
                            Dim oldTop As Single: oldTop = shape.top
                            Dim oldWidth As Single: oldWidth = shape.width
                            Dim oldHeight As Single: oldHeight = shape.height
                            Dim oldZOrder As Integer: oldZOrder = shapeZOrder(oldName & "-" & shape.ZOrderPosition)

                            shape.Delete

                            ' add new picture
                            Dim newShape As PowerPoint.shape
                            Set newShape = slide.Shapes.AddPicture(filePaths.Item(idx), _
                                    msoFalse, msoTrue, oldLeft, oldTop, oldWidth, oldHeight)

                            ' restore cropping
                            With newShape.PictureFormat
                                .CropTop = oldCropTop
                                .CropBottom = oldCropBottom
                                .CropLeft = oldCropLeft
                                .CropRight = oldCropRight
                            End With

                            ' Restore the ZOrder
                            Do While newShape.ZOrderPosition < oldZOrder
                                newShape.ZOrder msoBringForward
                            Loop
                            Do While newShape.ZOrderPosition > oldZOrder
                                newShape.ZOrder msoSendBackward
                            Loop

                            newShape.Name = oldName
                            
                            imagesReplaced = imagesReplaced + 1

                            ' sleep for 50 milliseconds
                            Sleep 50
                            'Debug.Print "----FILE WAS REPLACED ----- "
                        End If
                    ElseIf Right(fileNames.Item(idx), 3) = "txt" Then
                        If shape.Type = msoTextBox Then ' if shape is a text box
                            ' read the content of the text file
                            Dim textStream As textStream
                            Set textStream = fso.OpenTextFile(filePaths.Item(idx), ForReading)
                            Dim text As String
                            text = textStream.ReadAll
                            textStream.Close
                    
                            ' parse the text and format the text box content
                            Dim collection As collection
                            Set collection = ParseText(text)
                           
                            ApplyTextStyle shape.TextFrame.textRange, collection
                            shape.TextFrame2.textRange.Font.Allcaps = msoFalse
                            textsReplaced = textsReplaced + 1
                        ElseIf shape.Type = msoTable Then ' if shape is a table
                            ' read the content of the text file
                            Dim textStreamTab As textStream
                            Set textStreamTab = fso.OpenTextFile(filePaths.Item(idx), ForReading)
                            Dim textTab As String
                            textTab = textStreamTab.ReadAll
                            textStreamTab.Close
                        
                            ' parse the text and format the table content
                            Dim collectionTab As collection
                            Set collectionTab = ParseTableText(textTab)
                            Debug.Print textTab
                            Dim dict As Object
                            For Each dict In collectionTab
                                Dim row As Integer
                                row = dict("row")
                                Dim col As Integer
                                col = dict("col")
                                If row <= shape.Table.Rows.Count And col <= shape.Table.Columns.Count Then
                                    Dim cellText As String
                                    cellText = dict("text")
                                    Dim cellCollection As collection
                                    Set cellCollection = ParseText(cellText)
                                    ApplyTextStyle shape.Table.Cell(row, col).shape.TextFrame.textRange, cellCollection
                                    shape.Table.Cell(row, col).shape.TextFrame2.textRange.Font.Allcaps = msoFalse
                                End If
                            Next dict
                            tablesReplaced = tablesReplaced + 1
                            ' sleep for 50 milliseconds
                            Sleep 50
                        End If
                    
                    End If
                Else
                    ' No matching file found, log the object name
                    noFileMatched = noFileMatched + 1
                    logFile.WriteLine "No matching file found for object: " & shape.Name & "  Slide: " & slide.SlideNumber
                End If
            End If
        Next shape
    Next slide
    ' Write the totals to the log file
    logFile.WriteLine "-"
    logFile.WriteLine " "
    logFile.WriteLine "Stats:"
    logFile.WriteLine "Total unmatched assets: " & noFileMatched
    logFile.WriteLine "Total images replaced: " & imagesReplaced
    logFile.WriteLine "Total texts replaced: " & textsReplaced
    logFile.WriteLine "Total tables replaced: " & tablesReplaced
    logFile.Close
    
    
    ' Create message box with stats
    If noFileMatched = 0 Then
        MsgBox "Stats:" & vbCrLf & _
        "Success, all taged objects were matched with a file" & vbCrLf & _
        " " & vbCrLf & _
        "Total images replaced: " & imagesReplaced & vbCrLf & _
        "Total texts replaced: " & textsReplaced & vbCrLf & _
        "Total tables replaced: " & tablesReplaced, _
        vbOKOnly + vbInformation, "Process Finished"
    Else
        MsgBox "Stats:" & vbCrLf & _
        "Total unmatched assets: " & noFileMatched & vbCrLf & _
        "look at the logFile for more information " & vbCrLf & _
        " " & vbCrLf & _
        "Total images replaced: " & imagesReplaced & vbCrLf & _
        "Total texts replaced: " & textsReplaced & vbCrLf & _
        "Total tables replaced: " & tablesReplaced, _
        vbOKOnly + vbExclamation, "Process Finished with unmatched assets"
    End If
End Sub